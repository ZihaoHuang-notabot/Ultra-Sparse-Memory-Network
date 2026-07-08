import torch
import torch.nn.functional as F
from collections import namedtuple
import os
from fused_index import FusedLookup

batch_size = 8192
num_indices = 128**2*8
group_size = 4
reduce_dim = 128
embedding_dim = 288
multi_tucker_core=2

torch.random.manual_seed(0)
indices = torch.randint(0, num_indices * group_size, (batch_size, reduce_dim), device='cuda:0', dtype=torch.int32, requires_grad=False)
weight1 = torch.normal(0, 2, (num_indices+1, embedding_dim), dtype=torch.bfloat16, device='cuda:0')
weight2 = weight1.clone().detach()
scores1 = torch.normal(0, 2, (batch_size, reduce_dim, multi_tucker_core), dtype=torch.bfloat16, device='cuda:0')
scores2 = scores1.clone().detach()

weight1.requires_grad=True
scores1.requires_grad=True
weight2.requires_grad=True
scores2.requires_grad=True
vocab_size = weight1.shape[0] - 1
per_layer_vocab_size = weight1.shape[0] - 1
shift = 0
weight1_bf16 = weight1 if weight1.dtype == torch.bfloat16 else weight1.to(torch.bfloat16)
main_grad1 = torch.zeros_like(weight1, dtype=torch.float32)
out1 = FusedLookup.apply(indices, weight1_bf16, main_grad1, scores1.permute(2,0,1).contiguous(), 0, vocab_size, per_layer_vocab_size, shift, group_size, False)

out_rand_like = torch.rand_like(out1)
loss1=(out1 * out_rand_like).sum()
#loss1=out1
loss1.sum().backward()
print(weight1.grad)

padding_idx = None
max_norm = None
norm_type = 2.
scale_grad_by_freq = False
sparse = False
val_idxs = indices % num_indices    


value_num=num_indices
values_for_look_up=weight2
#(bs, knn, 2)
best_scores = scores2
bs = best_scores.shape[0]
best_indice_shuffled = indices
real_indice = best_indice_shuffled % value_num
group_indice = best_indice_shuffled // value_num
group_indice = group_indice.to(torch.int64)
print("group_indice", group_indice)
values = torch.nn.functional.embedding(real_indice, values_for_look_up)   # output shape: [bs, knn, vdim]
values = values.view(bs, reduce_dim, multi_tucker_core, -1)
values = (values * best_scores.unsqueeze(dim=-1)).view(bs, reduce_dim, -1)
output = torch.zeros((bs, group_size, embedding_dim),device=values.device, dtype=values.dtype)
output.scatter_add_(1, group_indice.unsqueeze(dim=-1).expand(-1,-1,embedding_dim), values)  # output shape: [bs, value_expand_time, vdim]
out2 = output
out2 = out2.view(batch_size, group_size*embedding_dim)
loss2 = (out2 * out_rand_like).sum()
#loss2=out2
loss2.sum().backward()
# print("out2", out2)
#import ipdb; ipdb.set_trace()
diff1=out1-out2
print("weight1.grad", scores1.grad)
print("weight2.grad", scores2.grad)
print("diff1", diff1.max(), diff1.min())
diff2=main_grad1.to(weight2.grad.dtype)-weight2.grad
print("diff2", diff2.max(), diff2.min())
diff3=scores1.grad-scores2.grad
print("diff3", diff3.max(), diff3.min())

