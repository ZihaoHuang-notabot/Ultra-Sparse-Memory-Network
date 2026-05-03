import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from collections import namedtuple

from fuse_ops.fused_index import XperfGlu
import os

batch_size = 8192
num_indices = 128**2*8
group_size = 4
reduce_dim = 64
embedding_dim = 288
per_layer_vocab_size = 128**2

torch.cuda.manual_seed(10)
indices = torch.randint(0, num_indices * group_size, (batch_size, group_size * reduce_dim), device='cuda:0', dtype=torch.int32, requires_grad=False)
p_input1 = torch.randn((batch_size, group_size * embedding_dim), dtype=torch.bfloat16, device="cuda:0")
p_input2 = p_input1.clone().detach()
weight1 = torch.randn((num_indices, embedding_dim), dtype=torch.bfloat16, device="cuda:0")
weight2 = weight1.clone().detach()


weight1.requires_grad = True
weight2.requires_grad = True
p_input1.requires_grad = True
p_input2.requires_grad = True

vocab_size = weight1.shape[0]
per_layer_vocab_size = weight1.shape[0]
shift = 0
weight1_bf16 = weight1 if weight1.dtype == torch.bfloat16 else weight1.to(torch.bfloat16)
main_grad1 = torch.zeros_like(weight1, dtype=torch.float32)
out1 = XperfGlu.apply(indices, weight1_bf16, main_grad1, p_input1, vocab_size, per_layer_vocab_size, shift, group_size, 0, False)
# print("out1", out1)
out_rand_like = torch.rand_like(out1)
loss1=(out1 * out_rand_like).sum()
loss1.backward()

pre_weights2 = F.embedding(indices % num_indices, weight2).float()
pre_scores2 = torch.zeros_like(indices).float()
tmp_score = torch.einsum('bgn,bkn->bkg',(p_input2.view(batch_size, group_size, -1).float(), pre_weights2.float()))  # bs, knn, group
group_id = indices // vocab_size
for i in range(group_size):
    mask = group_id == i
    pre_scores2[mask] = tmp_score[:,:,i][mask]
out2 = pre_scores2.to(torch.bfloat16)
# print("out2", out2)
loss2=(out2 * out_rand_like).sum()
loss2.backward()

diff1=out1-out2
print("diff1", diff1.max(), diff1.min())
diff2=main_grad1.to(weight2.grad.dtype)-weight2.grad
print("diff2", diff2.max(), diff2.min())
diff3=p_input1.grad-p_input2.grad
print("diff3", diff3.max(), diff3.min())
