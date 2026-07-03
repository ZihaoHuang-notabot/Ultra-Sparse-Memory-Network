import torch 
import os
USE_NPU = os.environ.get("USE_NPU") == "True"
try:
    import torch_npu
    USE_NPU = torch.npu.is_available()
except ImportError:
    pass

from functools import partial
import einops

from torch import nn
import numpy as np
from .model import LayerNormBase
import math
import wandb
import torch.nn.functional as F
from .memory_parallel import get_memory_layer_parallel_rank
class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)

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

class ParallelKeyQueryInnerProductNew(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, keys, bs, n_keys, head_num, nhead_share_query, tucker_rank, half):
        # at this point
        # keys (2 * self.tucker_rank, half, head_num * n_keys)
        # query (self.tucker_rank * 2, bs, half)
        # scores = torch.empty(2, bs, tucker_rank * head_num * n_keys, device=query.device, dtype=query.dtype)
        # scores = torch.empty(2 * tucker_rank, bs, head_num * n_keys, device=query.device, dtype=query.dtype)
        # for i in range(2 * tucker_rank):
        #     torch.matmul(query[i], keys[i], out=scores[i])
        scores = torch.matmul(query, keys)
        ctx.bs = bs
        ctx.n_keys = n_keys
        ctx.head_num = head_num
        ctx.tucker_rank = tucker_rank
        ctx.half = half
        ctx.nhead_share_query = nhead_share_query
        ctx.save_for_backward(query, keys)
        return scores
    
    @staticmethod
    def backward(ctx, score_grad):
        (query, keys) = ctx.saved_tensors
        #q1 grad
        #(m, k) = (m, n) (k, n) T
        query_grad = torch.matmul(score_grad, keys.to(score_grad.dtype).transpose(2,1))
        #kn = (mk)T (mn)
        keys_grad = torch.matmul(query.to(score_grad.dtype).transpose(2,1), score_grad)
        
        return query_grad, keys_grad, None, None, None, None, None, None

class UltraMemLayerV2(torch.nn.Module):
    def __init__(self, hidden_size, knum, kdim, vdim, pre_vdim, knn, head=1, tucker_rank=2, tucker_multihead=2, value_expand_time=4, tucker_rank_penalty=0, layer_id=0, has_value=False, share_ratio=1.0, mem_layer_num=1, config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.has_value = has_value

        q_proj_out_dim = kdim * 2 * tucker_rank if config.mem_q_for_each_tucker_rank else kdim * 2
        self.key_expand_time = config.mem_key_expand_time
        self.fake_value_expand_time = self.key_expand_time * self.key_expand_time

        # get shared value num and perlayer value num
        assert 0 < share_ratio <= 1
        assert value_expand_time == 1 or value_expand_time == self.fake_value_expand_time
        self.offset = layer_id * knum ** 2
        self.all_value_num = knum * knum * mem_layer_num
        knum = int((self.all_value_num * share_ratio) ** 0.5)
        value_num = knum ** 2
        virtual_value_num = self.fake_value_expand_time * value_num
        key_num = self.key_expand_time * knum

        v_proj_in_dim = vdim * value_expand_time

        print(f'UltraMemV2----layer_id:{layer_id}, real_knum:{knum}, offset:{self.offset}')


        self.kdim = kdim
        self.vdim = vdim
        self.pre_vdim = pre_vdim
        self.knn = knn
        self.final_top_expand_time = config.mem_final_top_expand_time
        self.tucker_rank = tucker_rank
        self.key_num = key_num
        self.tucker_multihead = tucker_multihead
        self.value_num = value_num
        self.value_expand_time = value_expand_time
        self.virtual_value_num = virtual_value_num
        self.head = head
        self.tucker_rank_penalty = tucker_rank_penalty
        self.v_proj_in_dim = v_proj_in_dim
        self.hidden_size = hidden_size
        self.key_balance_reg_coef = config.mem_key_balance_reg_coef
        self.mem_q_for_each_tucker_rank = config.mem_q_for_each_tucker_rank
        self.use_glu_act = config.mem_use_glu_act
        
        self.output_dropout = Dropout(config.mem_output_dropout_rate)
        self.vertical_parallel = config.vertical_parallel
        self.mem_parallel_size = config.mem_parallel_size
        if self.vertical_parallel:
            assert config.mem_parallel_size > 1
        else:
            assert config.mem_parallel_size == 1

        if self.has_value:
            assert vdim % self.mem_parallel_size == 0
            assert self.pre_vdim % self.mem_parallel_size == 0
            self.local_vdim = vdim // self.mem_parallel_size
            self.local_pre_vdim = self.pre_vdim // self.mem_parallel_size
            #random data in different rank
            if self.vertical_parallel:
                rank = get_memory_layer_parallel_rank()
                start = rank * self.local_vdim
            else:
                start = 0
            values_for_look_up = torch.randn(self.all_value_num, vdim)[:, start:start+self.local_vdim].contiguous()
            self.values_for_look_up = nn.Parameter(values_for_look_up)
            if self.pre_vdim > 0:
                if self.vertical_parallel:
                    rank = get_memory_layer_parallel_rank()
                    start = rank * self.local_pre_vdim
                else:
                    start = 0
                pre_values_for_look_up = torch.randn(self.all_value_num, self.pre_vdim)[:, start:start+self.local_pre_vdim].contiguous()
                self.pre_values_for_look_up = nn.Parameter(pre_values_for_look_up)
        else:
            self.query_proj = nn.Linear(hidden_size, q_proj_out_dim, bias=False)
            # self.keys = nn.Parameter(torch.randn(2, tucker_rank, kdim, head, key_num))
            self.keys = nn.Parameter(torch.randn(head, 2, key_num, kdim, tucker_rank))
            self.values_proj = nn.Linear(self.v_proj_in_dim, hidden_size, bias=False)
            if self.pre_vdim > 0 :
                self.pre_values_proj = nn.Linear(hidden_size, self.pre_vdim, bias=False)
            self.query_norm = LayerNormBase.build(config, size=kdim, elementwise_affine=config.attention_layer_norm_with_affine)
            self.keys_norm = LayerNormBase.build(config, size=kdim, elementwise_affine=config.attention_layer_norm_with_affine)
            self.tucker_core = nn.ParameterList([nn.Parameter(torch.randn(head, tucker_rank, tucker_rank)) for _ in range(tucker_multihead)])
            self.register_buffer("shuffle_index", torch.randperm(virtual_value_num))   #ShuffleIndex(virtual_value_num)
            self.register_buffer("tucker_core_u", torch.zeros([1, head, 1, tucker_rank]))
            self.register_buffer("tucker_core_v", torch.zeros([1, head, 1, tucker_rank]))

    def reset_parameters(self, distributed_strategy):
        if self.has_value:
            # value init
            std_value1 = 0.02
            if self.vertical_parallel:
                rank = get_memory_layer_parallel_rank()
                start = rank * self.local_vdim
            else:
                start = 0
            tmp = torch.normal(
                    mean=0, std=std_value1, size=(self.all_value_num, self.vdim),
                    device="cpu", dtype=self.values_for_look_up.dtype
                )
            nn.init.zeros_(self.values_for_look_up)
            with torch.no_grad():
                self.values_for_look_up.add_(
                        tmp[
                            :,
                            start:start+self.local_vdim,
                        ].to(self.values_for_look_up.device)
                    )
            if self.pre_vdim > 0:
                if self.vertical_parallel:
                    rank = get_memory_layer_parallel_rank()
                    start = rank * self.local_pre_vdim
                else:
                    start = 0
                tmp = torch.normal(
                        mean=0, std=std_value1, size=(self.all_value_num, self.pre_vdim),
                        device="cpu", dtype=self.pre_values_for_look_up.dtype
                    )
                nn.init.zeros_(self.pre_values_for_look_up)
                with torch.no_grad():
                    self.pre_values_for_look_up.add_(
                            tmp[
                                :,
                                start:start+self.local_pre_vdim,
                            ].to(self.pre_values_for_look_up.device)
                        )
        else:
            nn.init.constant_(self.query_norm.weight, val=1/self.kdim**0.4)
            nn.init.constant_(self.keys_norm.weight, val=1/self.kdim**0.4)

            nn.init.normal_(self.query_proj.weight, mean=0.0, std=1/(math.sqrt(self.kdim)))
            nn.init.normal_(self.values_proj.weight, mean=0.0, std=1/(math.sqrt(self.v_proj_in_dim)))

            nn.init.normal_(self.keys, mean=0.0, std=1/(math.sqrt(self.kdim)))
            if self.pre_vdim > 0 :
                nn.init.normal_(self.pre_values_proj.weight, mean=0.0, std=1/(math.sqrt(self.hidden_size)))


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

    def post_load_init(self):
        if USE_NPU:
            if not self.has_value:        
                self.tucker_core_uv = torch.nn.Parameter(torch.stack([self.tucker_core_u.flatten(), self.tucker_core_v.flatten()], dim=-1).contiguous())
                self.tucker_core_stacked = torch.nn.Parameter(torch.stack(tuple(self.tucker_core), dim=0).contiguous())
            else:
                self.pre_values_for_look_up.data = self.pre_values_for_look_up.data.to(torch.bfloat16)
                self.values_for_look_up.data = self.values_for_look_up.data.to(torch.bfloat16)
        
    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.post_load_init()
        return result
    
    def _calc_tucker_1rank(self):
        tucker_core_sum = torch.stack(list(self.tucker_core), dim=0).sum(dim=0).to(torch.float32)

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
        if self.has_value:
            return
        U, V, _ = self._calc_tucker_1rank()

        U,V = U.detach().to(self.tucker_core[0].dtype), V.detach().to(self.tucker_core[0].dtype)

        # must maintain smoothness
        pos_sim = torch.abs(U - self.tucker_core_u).sum(dim=-1,keepdim=True) + torch.abs(V - self.tucker_core_v).sum(dim=-1,keepdim=True)
        neg_sim = torch.abs(-U - self.tucker_core_u).sum(dim=-1,keepdim=True) + torch.abs(-V - self.tucker_core_v).sum(dim=-1,keepdim=True)
        sign = torch.sign(neg_sim - pos_sim)
        sign[sign == 0] = 1
        self.tucker_core_u, self.tucker_core_v = U * sign, V * sign
        self.tucker_core_uv = torch.stack([self.tucker_core_u, self.tucker_core_v], dim=0)

    def forward(self, hidden_state, full_memory_layer):
        prefix_shape = hidden_state.shape[:-1]
        bs = np.prod(prefix_shape)
        # _, _, aux_loss = self._calc_tucker_1rank()

        query = self.query_proj(hidden_state.view(bs, -1))   # output shape: [bs, q_proj_out_dim]
        qtr = self.tucker_rank if self.mem_q_for_each_tucker_rank else 1
        query = query.view(bs, 2, qtr, self.kdim)             # output shape: [bs, 2, kdim]
        query = self.query_norm(query)
        if not USE_NPU:
            query = query.view(bs, 2, qtr*self.kdim)

        keys = self.keys_norm(self.keys.transpose(3,4)).transpose(3,4)

        best_scores, best_indice, balance_reg_loss_key = self.TuckerDecomposedQueryKeyRetrieval(query, keys)

        # get real index
        if not USE_NPU:
            best_indice = self.shuffle_index[best_indice]
        else:
            best_indice = torch.nn.functional.embedding(best_indice, self.shuffle_index.unsqueeze(1)).squeeze(-1)
        #group_indice = best_indice // self.value_num
        #real_indice = ((best_indice % self.value_num) + self.offset) % self.all_value_num

        if self.pre_vdim > 0:
            pre_input = self.pre_values_proj(hidden_state.view(bs, -1))
        else:
            pre_input = None
        if self.vertical_parallel and self.mem_parallel_size > 1:
            output = full_memory_layer.ImplicitValueExpansionParallel(best_scores, best_indice, pre_input, self.all_value_num, self.value_num, self.offset)
        else:
            output = full_memory_layer.ImplicitValueExpansion(best_scores, best_indice, pre_input, self.all_value_num, self.value_num, self.offset)

        if self.value_expand_time != self.key_expand_time * self.key_expand_time:
            assert self.value_expand_time == 1
            output = output.view(bs, self.fake_value_expand_time, self.vdim)
            output = torch.sum(output, dim=1, keepdim=False)

        output = self.values_proj(output)

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
        if self.key_balance_reg_coef > 0 and balance_reg_loss_key is not None:
            output = AuxLossBackwardHook.apply(output, balance_reg_loss_key, self.key_balance_reg_coef)

        output = self.output_dropout(output)
        return output.view(*prefix_shape, -1)

    def TuckerDecomposedQueryKeyRetrieval(self, query, keys):
        bs = query.shape[0]
        if USE_NPU:
            qtr = query.shape[-2]
        head_num, _, n_keys, kdim, tucker_rank = keys.shape
        nhead_share_query = 2

        # generate score
        if not USE_NPU:
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
        else:
            assert nhead_share_query <= 2, "This optimization has only been implemented for nhead_share_query <= 2.  Otherwise, you need to replace every chunk (chunk.size=nhead_share_query) of nhead_share_query heads with the first head in that chunk to maintain correctness"
            # Note: Due to different contraction order in the einsum than the original, there is a slight error in scores_refine.  This may be eliminated by casting query and keys to float16 (not bfloat16!) first for the extra required mantissa bits
            scores_refine = einops.einsum(
                query.reshape(bs, 2, qtr, self.kdim)[:,:,0,:].unsqueeze(2),
                keys, 
                "bs tensor_rank heads kdim, heads tensor_rank n_keys kdim tucker_rank -> bs heads n_keys tucker_rank tensor_rank"
            )
            scores1_refine, scores2_refine = map(partial(torch.squeeze, dim=-1), scores_refine.split(1, dim=-1))

        if not USE_NPU:
            scores1 = (scores1_refine * self.tucker_core_u).sum(-1)#.detach()
            scores2 = (scores2_refine * self.tucker_core_v).sum(-1)#.detach()
        else:
            scores = einops.einsum(
                query.view(bs, 2, qtr, self.kdim)[:,:,0,:].unsqueeze(2),
                keys, 
                self.tucker_core_uv,
                "bs tensor_rank heads kdim, heads tensor_rank n_keys kdim tucker_rank, tucker_rank tensor_rank -> tensor_rank bs heads n_keys"
            )
            scores1, scores2 = map(partial(torch.squeeze, dim=0), scores.split(1, dim=0))

        scores1_chosen, indices1 = scores1.topk(self.knn, dim=2, largest=True, sorted=True)
        scores2_chosen, indices2 = scores2.topk(self.knn, dim=2, largest=True, sorted=True)


        # blc loss
        if self.key_balance_reg_coef > 0:
            p1 = scores1.float().softmax(dim=-1).view(-1,self.key_num).sum(dim=0) / (bs*head_num)
            index_counts = torch.zeros(self.key_num, dtype=torch.float32, device=scores1.device)
            index_counts.scatter_add_(0, indices1.flatten(), torch.ones_like(indices1.flatten(), dtype=torch.float32, device=scores1.device))
            f1 = index_counts / index_counts.sum()
            balance_reg_loss1 = self.key_num * (p1 * f1).sum()

            p2 = scores2.float().softmax(dim=-1).view(-1,self.key_num).sum(dim=0) / (bs*head_num)
            index_counts = torch.zeros(self.key_num, dtype=torch.float32, device=scores2.device)
            index_counts.scatter_add_(0, indices2.flatten(), torch.ones_like(indices2.flatten(), dtype=torch.float32, device=scores2.device))
            f2 = index_counts / index_counts.sum()
            balance_reg_loss2 = self.key_num * (p2 * f2).sum()

            balance_reg_loss_key = balance_reg_loss1 + balance_reg_loss2
        else:
            balance_reg_loss_key = None



        _scores1_refine = scores1_refine.gather(2, indices1.unsqueeze(dim=-1).expand(-1,-1,-1,scores1_refine.shape[-1]))
        _scores2_refine = scores2_refine.gather(2, indices2.unsqueeze(dim=-1).expand(-1,-1,-1,scores2_refine.shape[-1])) # [bs, head_num, topk, tucker_rank]

        if not USE_NPU:
            score_list = []
            for i in range(self.tucker_multihead):
                score_list.append((_scores1_refine @ (self.tucker_core[i]) @ _scores2_refine.transpose(-1,-2)).view(bs, head_num, -1))
            all_scores = torch.stack(score_list, dim=-1).sum(dim=-1)
        else:
            score_list_mat = einops.einsum(
                _scores1_refine, 
                self.tucker_core_stacked,
                _scores2_refine, 
                "bs head n_keys1 tucker_rank1, core_head head tucker_rank1 tucker_rank2, bs head n_keys2 tucker_rank2 -> core_head bs head n_keys1 n_keys2"
            )
            score_list = score_list_mat.reshape(self.tucker_multihead, bs, head_num, -1)
            all_scores = score_list.sum(dim=0)

        if not USE_NPU:
            all_indices = (
                indices1.view(bs, head_num, self.knn, 1).expand(bs, head_num, self.knn, self.knn) * n_keys +
                indices2.view(bs, head_num, 1, self.knn).expand(bs, head_num, self.knn, self.knn)
            ).view(bs, head_num, -1)
        else:
            all_indices = (
                indices1.view(bs, head_num, self.knn, 1) * n_keys +
                indices2.view(bs, head_num, 1, self.knn)
            ).view(bs, head_num, -1)

        
        scores, best_indices = torch.topk(all_scores, k=self.knn, dim=2, largest=True, sorted=True)

        best_scores = torch.stack([s.gather(2, best_indices) for s in score_list], dim=-1).view(bs,-1,self.tucker_multihead)
        best_indices = all_indices.gather(2, best_indices).view(bs,-1)

        return best_scores, best_indices, balance_reg_loss_key

    def ImplicitValueExpansionParallel(self, best_scores, best_indice, pre_input, all_value_num, value_num, offset):
        if not USE_NPU:  
            from fuse_ops.fused_index import XperfGlu, FusedLookup
        else:  # NPU does not support CUDA kernels
            from .pytorch_kernels import pytorch_xperf_glu as XperfGlu, pytorch_fused_lookup as FusedLookup
            XperfGlu.apply = XperfGlu
            FusedLookup.apply = FusedLookup

        from .memory_parallel import gather_from_memory_layer_parallel_region, AllReduceInMemGroup, tucker_multihead_gather_score, get_memory_layer_parallel_world_size, EmbeddingAll2AllSingle, get_memory_layer_parallel_group
        best_indice = best_indice.to(torch.int32)
        entry_num = best_indice.shape[0]
        token_per_entry = best_indice.shape[1]
        memory_layer_parallel_size = get_memory_layer_parallel_world_size()
        memory_layer_parallel_group = get_memory_layer_parallel_group()
        token_count_buffer = torch.empty([memory_layer_parallel_size], dtype=torch.int32, device=best_indice.device)
        torch.distributed.all_gather_into_tensor(token_count_buffer,
                                                torch.tensor(best_indice.numel(), dtype=torch.int32, device=best_indice.device),
                                                group=memory_layer_parallel_group)
        # get max, already multiples of token_per_entry
        max_token_num = int(token_count_buffer.max())
        has_padding_idx = True
        padding_idx = all_value_num
        max_entry_num = max_token_num // token_per_entry
        best_indice = best_indice.view(-1)
        if best_indice.numel() < max_token_num:
            if not USE_NPU:
                best_indice = F.pad(best_indice, (0, max_token_num - best_indice.numel()), value=padding_idx)
            else:
                pad_size = max_token_num - best_indice.numel()
                random_padding = torch.randint(0, max_token_num, (pad_size,), dtype=best_indice.dtype, device=best_indice.device)
                best_indice = torch.cat([best_indice, random_padding])
        else:
            best_indice = best_indice

        total_indices = gather_from_memory_layer_parallel_region(best_indice)
        total_indices = total_indices.view(-1, token_per_entry)

        if self.tucker_multihead > 1:
            best_score = best_scores.permute(2,0,1).contiguous()
            new_scores = [best_score[0], best_score[1]]
            for i in range(len(new_scores)):
                new_scores[i] = new_scores[i].view(-1)
                if new_scores[i].numel() < max_token_num:
                    new_scores[i] = F.pad(new_scores[i], (0, max_token_num - new_scores[i].numel()), value=0)
            all_score = tucker_multihead_gather_score(*new_scores)
        else:
            all_score = gather_from_memory_layer_parallel_region(best_scores)
        all_score = all_score.view(-1, token_per_entry)

        if pre_input is not None:
            bs = pre_input.shape[0]
            if bs < max_entry_num:
                pre_input = F.pad(pre_input.view(-1), (0, max_entry_num*pre_input.shape[1] - pre_input.numel()), value=0)
            else:
                pre_input = pre_input.view(-1)
            send_receive_count = [max_entry_num * self.local_pre_vdim for i in range(memory_layer_parallel_size)]
            outputs_buffer = torch.empty(max_entry_num * self.local_pre_vdim * memory_layer_parallel_size, dtype=pre_input.dtype, device=pre_input.device, requires_grad=False)
            p_inputs = pre_input.view(max_entry_num, memory_layer_parallel_size, self.local_pre_vdim).permute(1,0,2).contiguous()
            output_parallel = [p_inputs.view(-1), outputs_buffer, send_receive_count, send_receive_count]
            output = EmbeddingAll2AllSingle.apply(*output_parallel)
            output = output.view(memory_layer_parallel_size * max_entry_num, self.local_pre_vdim)
            pre_score = XperfGlu.apply(total_indices, self.pre_values_for_look_up, output, all_value_num, value_num, offset, 1, 0, False)
            pre_score = AllReduceInMemGroup().apply(pre_score)
            if self.use_glu_act:
                pre_score = F.gelu(pre_score)
            all_score = pre_score.squeeze(-1) * all_score
        emb = FusedLookup.apply(total_indices, self.values_for_look_up, all_score, padding_idx, all_value_num, value_num, offset, self.fake_value_expand_time, has_padding_idx)
        entry_counts = [max_entry_num] * memory_layer_parallel_size
        emb_send_recv_count = [x * emb.shape[-1] for x in entry_counts]
        embedding_outputs_buffer = torch.empty(max_entry_num * emb.shape[-1] * memory_layer_parallel_size, dtype=emb.dtype, device=emb.device, requires_grad=False)
        output_parallel = [emb.view(-1), embedding_outputs_buffer, emb_send_recv_count, emb_send_recv_count]
        output = EmbeddingAll2AllSingle.apply(*output_parallel)
        num_group = self.fake_value_expand_time
        output = output.view(memory_layer_parallel_size, max_entry_num * num_group, emb.shape[-1] // num_group).transpose(0, 1)
        concated_output = output.reshape(max_entry_num, memory_layer_parallel_size * emb.shape[-1])
        return concated_output[0:entry_num]

    def ImplicitValueExpansion(self, best_scores, best_indice, pre_input, all_value_num, value_num, offset):
        if not USE_NPU:  # NPU does not support CUDA kernels
            from fuse_ops.fused_index import XperfGlu, FusedLookup
            if pre_input is not None:
                pre_score = XperfGlu.apply(best_indice.to(torch.int32), self.pre_values_for_look_up, pre_input, all_value_num, value_num, offset, 1, 0, False)
                if self.use_glu_act:
                    pre_score = F.gelu(pre_score)
                best_scores = pre_score.unsqueeze(-1) * best_scores
            if self.tucker_multihead > 1:
                best_scores = best_scores.permute(2,0,1).contiguous()
            else:
                best_scores = best_scores.squeeze(dim=-1)
            output = FusedLookup.apply(best_indice.to(torch.int32), self.values_for_look_up, best_scores, 0, all_value_num, value_num, offset, self.fake_value_expand_time, False)
        else:
            real_indice = ((best_indice % value_num) + offset) % all_value_num

            bs = best_scores.shape[0]
            values = torch.nn.functional.embedding(real_indice, self.values_for_look_up)   # output shape: [bs, knn, vdim]
            values = values.view(bs, self.knn*self.head, self.tucker_multihead, -1)

            if pre_input is not None:
                pre_values = torch.nn.functional.embedding(real_indice, self.pre_values_for_look_up)   # output shape: [bs, knn, pre_vdim]
                pre_scores = torch.einsum('bd,bkd->bk',(pre_input, pre_values))
                values = (values * (best_scores.unsqueeze(dim=-1) * pre_scores[...,None,None])).view(bs, self.knn*self.head, -1)
            else:
                values = (values * best_scores.unsqueeze(dim=-1)).view(bs, self.knn*self.head, -1)

            if self.value_expand_time == 1:
                output = values.sum(dim=1)
            else:
                from fuse_ops.scatter_add import ScatterAdd
                output = ScatterAdd.apply(group_indice, values, self.value_expand_time)
            output = output.view(bs, -1)
        return output
