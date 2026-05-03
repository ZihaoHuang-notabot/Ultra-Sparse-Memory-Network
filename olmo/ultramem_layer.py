import torch
from torch import nn
import numpy as np
from .model import LayerNormBase
import math
import wandb

class AuxLossBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, aux_loss, scale):
        # Preserve the aux_loss by storing it in the context to avoid garbage collection.
        ctx.save_for_backward(aux_loss)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Scale the auxiliary loss like the main loss.
        aux_loss, = ctx.saved_tensors
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * ctx.scale
        return grad_output, scaled_aux_loss_grad, None

class ParallelKeyQueryInnerProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, keys, bs, n_keys, head_num, nhead_share_query):
        """
          equal to   
              scores1 = torch.matmul(q1.transpose(0, 1), keys[:, 0].transpose(1,2)) # (head, bs*seq_len, num_keys)
              scores2 = torch.matmul(q2.transpose(0, 1), keys[:, 1].transpose(1,2)) # (head, bs*seq_len, num_keys)
          args: 
            - q1 [bs, head, k_dim/2]
            - q2 [bs, head, k_dim/2]
            - keys [head, 2, num_keys, k_dim/2]
        """
        k_dim = query.shape[-1]
        half = k_dim // 2

        scores1 = torch.zeros(bs, head_num, n_keys, device=query.device, dtype=query.dtype) # (head, bs, num_keys)
        scores2 = torch.zeros(bs, head_num, n_keys, device=query.device, dtype=query.dtype) # (head, bs, num_keys)

        for i in range(head_num):
            q1 = query[:, i // nhead_share_query, :half]                                                                                          # (bs, half)
            q2 = query[:, i // nhead_share_query, half:]
            key1 = keys[i, 0]
            key2 = keys[i, 1]
            torch.matmul(q1, key1.transpose(0,1), out=scores1[:, i])
            torch.matmul(q2, key2.transpose(0,1), out=scores2[:, i])

        ctx.bs = bs
        ctx.n_keys = n_keys
        ctx.head_num = head_num
        ctx.half = half
        ctx.nhead_share_query = nhead_share_query
        ctx.save_for_backward(query, keys)
        return scores1, scores2
    
    @staticmethod
    def backward(ctx, score1_grad, score2_grad):
        (query, keys) = ctx.saved_tensors
        keys_grad = torch.zeros_like(keys)
        query_grad = torch.zeros_like(query)

        # 更多的矩阵, 多 element_wise_add
        for i in range(ctx.head_num):
            torch.matmul(score1_grad[:, i].transpose(0, 1), query[:, i // ctx.nhead_share_query, :ctx.half], out=keys_grad[i][0])
            torch.matmul(score2_grad[:, i].transpose(0, 1), query[:, i // ctx.nhead_share_query, ctx.half:], out=keys_grad[i][1])
            q1_g = torch.matmul(score1_grad[:, i], keys[i][0])
            q2_g = torch.matmul(score2_grad[:, i], keys[i][1])
            query_grad[:, i // ctx.nhead_share_query, :ctx.half].add_(q1_g)
            query_grad[:, i // ctx.nhead_share_query, ctx.half:].add_(q2_g)
        return query_grad, keys_grad, None, None, None, None

class UltraMemLayerV1(torch.nn.Module):
    def __init__(self, hidden_size, knum, kdim, vdim, knn, head=1, tucker_rank=2, tucker_multihead=2, value_expand_time=4, tucker_rank_penalty=0, layer_id=0, config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        q_proj_out_dim = kdim * 2
        v_proj_in_dim = vdim * value_expand_time
        key_expand_time = int(value_expand_time ** 0.5)
        value_num = knum * knum
        virtual_value_num =  value_expand_time * value_num
        key_num = key_expand_time * knum

        self.kdim = kdim
        self.vdim = vdim
        self.knn = knn
        self.tucker_rank = tucker_rank
        self.key_num = key_num
        self.tucker_multihead = tucker_multihead
        self.value_num = value_num
        self.value_expand_time = value_expand_time
        self.virtual_value_num = virtual_value_num
        self.head = head
        self.tucker_rank_penalty = tucker_rank_penalty
        self.v_proj_in_dim = v_proj_in_dim

        self.query_proj = nn.Linear(hidden_size, q_proj_out_dim, bias=False)
        self.keys = nn.Parameter(torch.randn(head, 2, key_num, kdim, tucker_rank))

        self.values_for_look_up = nn.Parameter(torch.randn(value_num, vdim))
        self.values_proj = nn.Linear(v_proj_in_dim, hidden_size, bias=False)
        self.query_norm = LayerNormBase.build(config, size=kdim, elementwise_affine=config.attention_layer_norm_with_affine)
        self.keys_norm = LayerNormBase.build(config, size=kdim, elementwise_affine=config.attention_layer_norm_with_affine)
        self.tucker_core = nn.ParameterList([nn.Parameter(torch.randn(head, tucker_rank, tucker_rank)) for _ in range(tucker_multihead)])
        self.register_buffer("shuffle_index", torch.randperm(virtual_value_num))   #ShuffleIndex(virtual_value_num)
        self.register_buffer("tucker_core_u", torch.zeros([1, head, 1, tucker_rank]))
        self.register_buffer("tucker_core_v", torch.zeros([1, head, 1, tucker_rank]))

    def reset_parameters(self, distributed_strategy):
        nn.init.constant_(self.query_norm.weight, val=1/self.kdim**0.33)
        nn.init.constant_(self.keys_norm.weight, val=1/self.kdim**0.33)

        nn.init.normal_(self.query_proj.weight, mean=0.0, std=1/(math.sqrt(self.kdim)))
        nn.init.normal_(self.values_proj.weight, mean=0.0, std=1/(math.sqrt(self.v_proj_in_dim)))

        # key init
        nn.init.normal_(self.keys, mean=0.0, std=1/(math.sqrt(self.kdim)))

        # value init
        std_value = 0.02
        nn.init.normal_(self.values_for_look_up, mean=0.0, std=std_value)

        # tucker core init
        for c in self.tucker_core:
            nn.init.uniform_(c, a=0.0, b=1)

        # shuffle index init
        shuffle_idx = torch.randperm(self.virtual_value_num, device=self.shuffle_index.device)
        from olmo.config import DistributedStrategy
        if distributed_strategy == DistributedStrategy.fsdp:
            torch.distributed.broadcast(shuffle_idx, src=0)
        self.shuffle_index.data = shuffle_idx
        self.tucker_core_u.data = torch.zeros_like(self.tucker_core_u, device=self.tucker_core_u.device)
        self.tucker_core_v.data = torch.zeros_like(self.tucker_core_v, device=self.tucker_core_v.device)
        self.tucker_core_uv = torch.stack([self.tucker_core_u, self.tucker_core_v], dim=0)
        self.indice_for_log = None
        self.output_std = None
        self.score_top1_mean = None
        self.score_topn_mean = None

    
    def _calc_tucker_1rank(self):
        tucker_core_sum = torch.stack(list(self.tucker_core), dim=0).sum(dim=0)

        U, S, V = torch.svd(tucker_core_sum)
        u = U[..., 0]  # [rank]
        v = V[..., 0]
        u = u[None,:,None,:]
        v = v[None,:,None,:]
        aux_loss = (torch.nn.functional.relu(S[...,1:] - 0.15) ** 2).mean()
        if wandb.run is not None and wandb.run.step!=0 and wandb.run.step % 10 == 0 and self.training:
            tmp_S = S.detach().mean(dim=0).cpu().tolist()
            wandb.log({f'ultramem/tucker_rank0/mem_layer_{self.layer_id:03d}':tmp_S[0],f'ultramem/tucker_rank1/mem_layer_{self.layer_id:03d}':tmp_S[1]}, step=wandb.run.step)
        return u, v, aux_loss
    
    def _online_update_tucker_approx(self):
        U, V, _ = self._calc_tucker_1rank()

        U,V = U.detach().to(self.tucker_core[0].dtype), V.detach().to(self.tucker_core[0].dtype)

        # must maintain smoothness
        pos_sim = torch.abs(U - self.tucker_core_u).sum(dim=-1,keepdim=True) + torch.abs(V - self.tucker_core_v).sum(dim=-1,keepdim=True)
        neg_sim = torch.abs(-U - self.tucker_core_u).sum(dim=-1,keepdim=True) + torch.abs(-V - self.tucker_core_v).sum(dim=-1,keepdim=True)
        sign = torch.sign(neg_sim - pos_sim)
        sign[sign == 0] = 1
        self.tucker_core_u, self.tucker_core_v = U * sign, V * sign
        self.tucker_core_uv = torch.stack([self.tucker_core_u, self.tucker_core_v], dim=0)

    def forward(self, hidden_state):
        prefix_shape = hidden_state.shape[:-1]
        bs = np.prod(prefix_shape)
        # _, _, aux_loss = self._calc_tucker_1rank()

        query = self.query_proj(hidden_state.view(bs, -1))   # output shape: [bs, q_proj_out_dim]
        query = query.view(bs, 1, 2, self.kdim)             # output shape: [bs, 2, kdim]
        query = self.query_norm(query)
        query = query.view(bs, 1, 2*self.kdim)

        keys = self.keys_norm(self.keys.transpose(-1,-2)).transpose(-1,-2)

        best_scores, best_indice = self.TuckerDecomposedQueryKeyRetrieval(query, keys)
        output = self.ImplicitValueExpansion(best_scores, best_indice)

        if wandb.run is not None and wandb.run.step!=0 and wandb.run.step % self.config.mem_log_interval == 0 and self.training:
            unique_idx_all, counts_all = best_indice.unique(return_counts=True)
            tmp_count_all = torch.zeros(self.virtual_value_num, dtype=counts_all.dtype, device=best_indice.device)
            tmp_count_all[unique_idx_all] = counts_all  
            if self.indice_for_log is None:
                self.indice_for_log = tmp_count_all
            else:
                self.indice_for_log += tmp_count_all
            
            self.output_std = output.std()
            self.score_top1_mean = best_scores[:,0].mean()
            self.score_topn_mean = best_scores[:,-1].mean()

        # if self.tucker_rank_penalty > 0 and aux_loss is not None:
        #     output = AuxLossBackwardHook.apply(output, aux_loss, self.tucker_rank_penalty)

        return output.view(*prefix_shape, -1)
    
    def TuckerDecomposedQueryKeyRetrieval(self, query, keys):
        bs = query.shape[0]
        head_num, _, n_keys, kdim, tucker_rank = keys.shape
        nhead_share_query = 2

        # generate score
        scores1_refine = []
        scores2_refine = []
        for key_idx in range(tucker_rank):
            scores1_refine_, scores2_refine_ = ParallelKeyQueryInnerProduct.apply(
                query, torch.select(keys,-1,key_idx), bs, n_keys, head_num, nhead_share_query
            )
            scores1_refine.append(scores1_refine_)
            scores2_refine.append(scores2_refine_)
        scores1_refine = torch.stack(scores1_refine, dim=-1)
        scores2_refine = torch.stack(scores2_refine, dim=-1)

        scores1 = (scores1_refine * self.tucker_core_u).sum(-1)#.detach()
        scores2 = (scores2_refine * self.tucker_core_v).sum(-1)#.detach()

        scores1_chosen, indices1 = scores1.topk(self.knn, dim=2, largest=True, sorted=True)
        scores2_chosen, indices2 = scores2.topk(self.knn, dim=2, largest=True, sorted=True)

        _scores1_refine = scores1_refine.gather(2, indices1.unsqueeze(dim=-1).expand(-1,-1,-1,scores1_refine.shape[-1]))
        _scores2_refine = scores2_refine.gather(2, indices2.unsqueeze(dim=-1).expand(-1,-1,-1,scores2_refine.shape[-1])) # [bs, head_num, topk, tucker_rank]

        score_list = []
        for i in range(self.tucker_multihead):
            score_list.append((_scores1_refine @ (self.tucker_core[i]) @ _scores2_refine.transpose(-1,-2)).view(bs, head_num, -1))
        all_scores = torch.stack(score_list, dim=-1).sum(dim=-1)

        all_indices = (
            indices1.view(bs, head_num, self.knn, 1).expand(bs, head_num, self.knn, self.knn) * n_keys +
            indices2.view(bs, head_num, 1, self.knn).expand(bs, head_num, self.knn, self.knn)
        ).view(bs, head_num, -1)
        scores, best_indices = torch.topk(all_scores, k=self.knn, dim=2, largest=True, sorted=True)

        best_scores = torch.stack([s.gather(2, best_indices) for s in score_list], dim=-1).view(bs,-1,self.tucker_multihead)
        best_indices = all_indices.gather(2, best_indices).view(bs,-1)

        return best_scores, best_indices

    def ImplicitValueExpansion(self, best_scores, best_indice):
        bs = best_scores.shape[0]
        best_indice_shuffled = self.shuffle_index[best_indice]
        if True:
            from fuse_ops.fused_index import FusedLookup
            if self.tucker_multihead > 1:
                best_scores = best_scores.permute(2,0,1).contiguous()
            else:
                best_scores = best_scores.squeeze(dim=-1)
            val_w = self.values_for_look_up
            val_w_bf16 = val_w if val_w.dtype == torch.bfloat16 else val_w.to(torch.bfloat16)
            output = FusedLookup.apply(best_indice_shuffled.to(torch.int32), val_w_bf16, val_w.main_grad, best_scores, 0, self.value_num, self.value_num, 0, self.value_expand_time, False)
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
        output = self.values_proj(output)
        return output