import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .model import LayerNormBase
import math
import wandb

class MemoryLayerPlus(torch.nn.Module):
    def __init__(self, hidden_size, knum, kdim, vdim, knn, head=1, layer_id=0, has_value=False, config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.has_value = has_value

        q_proj_out_dim = kdim * 2 * head

        # get shared value num and perlayer value num
        value_num = knum ** 2

        print(f'Memory+ layer----layer_id:{layer_id}, real_knum:{knum}')


        self.kdim = kdim
        self.vdim = vdim
        self.knn = knn
        self.key_num = knum
        self.value_num = value_num
        self.head = head
        self.hidden_size = hidden_size
        self.virtual_value_num = value_num
        assert vdim == hidden_size


        if self.has_value:
            self.values_for_look_up = nn.Parameter(torch.randn(self.value_num, vdim))
        else:
            self.query_proj = nn.Linear(hidden_size, q_proj_out_dim, bias=False)
            self.keys = nn.Parameter(torch.randn(head, 2, knum, kdim))
            self.values_proj = nn.Linear(vdim, hidden_size, bias=False)
            self.swilu_projection = nn.Linear(hidden_size, vdim, bias=False)

            self.query_norm = LayerNormBase.build(config, size=kdim, elementwise_affine=config.attention_layer_norm_with_affine)
            self.keys_norm = LayerNormBase.build(config, size=kdim, elementwise_affine=config.attention_layer_norm_with_affine)

    def reset_parameters(self, distributed_strategy):
        if self.has_value:
            # value init
            nn.init.normal_(self.values_for_look_up, mean=0.0, std=1/self.vdim**0.5)
        else:
            nn.init.xavier_uniform_(self.query_proj.weight)
            nn.init.normal_(self.values_proj.weight, mean=0.0, std=1/(math.sqrt(self.vdim)))
            nn.init.normal_(self.swilu_projection.weight, mean=0.0, std=1/(math.sqrt(self.vdim)))

            # key init
            bound = 1 / math.sqrt(self.kdim*2)
            nn.init.uniform_(self.keys, a=-bound, b=bound)

            nn.init.constant_(self.query_norm.weight, val=1.0)
            nn.init.constant_(self.keys_norm.weight, val=1.0)

            self.indice_for_log = None
            self.output_std = None
            self.score_top1_mean = None
            self.score_topn_mean = None

    
    def _online_update_tucker_approx(self):
        pass

    def forward(self, hidden_state, full_memory_layer):
        prefix_shape = hidden_state.shape[:-1]
        bs = np.prod(prefix_shape)
        knn = self.knn
        # _, _, aux_loss = self._calc_tucker_1rank()

        query = self.query_proj(hidden_state.view(bs, -1))   # output shape: [bs, q_proj_out_dim]
        query = query.view(bs, self.head, 2, self.kdim)             # output shape: [bs, 2, kdim]
        query = self.query_norm(query)
        query = query.view(-1, self.kdim*2)             # output shape: [bs*head, 2*kdim]

        scores, indices = self.get_indices(query, knn)  # (bs * heads, knn) ** 2

        # re-scoring
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs * heads, knn)

        # merge heads / knn (since we sum heads)
        indices = indices.view(bs, self.head * knn)  # (bs, heads * knn)
        scores = scores.view(bs, self.head * knn)  # (bs, heads * knn)

        output = self.get_output(hidden_state.view(bs, -1), scores, indices, full_memory_layer)

        if wandb.run is not None and wandb.run.step!=0 and wandb.run.step % self.config.mem_log_interval == 0 and self.training:
            unique_idx_all, counts_all = indices.unique(return_counts=True)
            tmp_count_all = torch.zeros(self.value_num, dtype=counts_all.dtype, device=indices.device)
            tmp_count_all[unique_idx_all] = counts_all  
            if self.indice_for_log is None:
                self.indice_for_log = tmp_count_all
            else:
                self.indice_for_log += tmp_count_all
            
            self.output_std = output.std()
            self.score_top1_mean = scores[:,0].mean()
            self.score_topn_mean = scores[:,-1].mean()

        # if self.tucker_rank_penalty > 0 and aux_loss is not None:
        #     output = AuxLossBackwardHook.apply(output, aux_loss, self.tucker_rank_penalty)

        return output.view(*prefix_shape, -1)
    
    def get_indices(self, query, knn):
        assert query.dim() == 2 and query.size(1) == self.kdim*2
        bs = len(query) // self.head
        query = query.view(-1, self.head, self.kdim*2)
        half = self.kdim
        # keys : (heads, 2, n_keys, half)
        # keys1 : (heads, n_keys, half)
        keys = self.keys_norm(self.keys)
        keys = self.keys.view(self.head, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = len(keys[0][0])

        # split query for product quantization
        q1 = query[:, :, :half]  # (bs, heads, half)
        q2 = query[:, :, half:]  # (bs, heads, half)

        # compute indices with associated scores
        scores1 = torch.einsum(
            "blh, lkh->blk", q1, keys1
        )  # (bs , heads, n_keys ** 0,5)
        scores2 = torch.einsum(
            "blh, lkh->blk", q2, keys2
        )  # (bs , heads, n_keys ** 0,5)

        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)  # (bs, heads, knn)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)  # (bs, heads, knn)

        # cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, self.head, knn, 1).expand(bs, self.head, knn, knn)
            + scores2.view(bs, self.head, 1, knn).expand(bs, self.head, knn, knn)
        ).view(
            bs, self.head, -1
        )  # (bs, heads, knn ** 2)
        all_indices = (
            indices1.view(bs, self.head, knn, 1).expand(bs, self.head, knn, knn)
            * n_keys
            + indices2.view(bs, self.head, 1, knn).expand(bs, self.head, knn, knn)
        ).view(
            bs, self.head, -1
        )  # (bs, heads, knn ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(
            all_scores, k=knn, dim=2, largest=True, sorted=True
        )  # (bs, heads, knn)
        indices = all_indices.gather(2, best_indices)  # (bs, knn)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, self.head, knn)
        return scores.view(bs * self.head, knn), indices.view(bs * self.head, knn)

    def get_output(self, input, best_scores, best_indice, full_memory_layer):
        if True:
            from fuse_ops.fused_index import FusedLookup
            best_scores = best_scores.squeeze(dim=-1)
            val_w = full_memory_layer.values_for_look_up
            val_w_bf16 = val_w if val_w.dtype == torch.bfloat16 else val_w.to(torch.bfloat16)
            output = FusedLookup.apply(best_indice.to(torch.int32), val_w_bf16, val_w.main_grad, best_scores, 0, self.value_num, self.value_num, 0, 1, False)
        else:
            real_indice = best_indice_shuffled % self.value_num
            group_indice = best_indice_shuffled // self.value_num

            values = torch.nn.functional.embedding(real_indice, self.values_for_look_up)   # output shape: [bs, knn, vdim]
            values = values.view(bs, self.knn*self.head, self.tucker_multihead, -1)
            values = (values * best_scores.unsqueeze(dim=-1)).view(bs, self.knn*self.head, -1)

            #replace torch.scatter_add with cuda kernel
            #output = torch.zeros((bs, self.value_expand_time, self.vdim),device=values.device, dtype=values.dtype)
            #output.scatter_add_(1, group_indice.unsqueeze(dim=-1).expand(-1,-1,self.vdim), values)  # output shape: [bs, value_expand_time, vdim]
            from fuse_ops.scatter_add import ScatterAdd
            output = ScatterAdd.apply(group_indice, values, self.value_expand_time)
            output = output.view(bs, -1)
        output = self.values_proj(output * F.silu(self.swilu_projection(input)))
        return output