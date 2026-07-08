import torch
import triton
import triton.language as tl

@triton.jit
def gather_indices_2d_kernel(indices1_ptr, indices2_ptr, best_indices_ptr, indices_ptr, head_num, n_keys, num_indices, knn: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_index = tl.program_id(2)

    offset_best_indices = pid_batch * head_num * num_indices + pid_head * num_indices + pid_index
    best_index = tl.load(best_indices_ptr + offset_best_indices)

    k1 = best_index // knn
    k2 = best_index % knn

    indices1_index = pid_batch * head_num * knn + pid_head * knn + k1
    indices2_index = pid_batch * head_num * knn + pid_head * knn + k2

    indices1_val = tl.load(indices1_ptr + indices1_index)
    indices2_val = tl.load(indices2_ptr + indices2_index)

    all_indices_val = indices1_val * n_keys + indices2_val

    offset_indices = pid_batch * head_num * num_indices + pid_head * num_indices + pid_index

    tl.store(indices_ptr + offset_indices, all_indices_val)


def triton_gather_indices_2d(indices1, indices2, best_indices, n_keys):
    bs, head_num, knn = indices1.shape
    num_indices = best_indices.shape[-1]

    indices = torch.empty((bs, head_num, num_indices), dtype=indices1.dtype, device=indices1.device)

    grid = (bs, head_num, num_indices)
    gather_indices_2d_kernel[grid](indices1, indices2, best_indices, indices, head_num, n_keys, num_indices, knn)

    return indices

@triton.jit
def einsum_forward(score_list, all_scores, scores1, scores2, tucker_core, knn: tl.constexpr, knn_align: tl.constexpr, tucker_rank: tl.constexpr, tucker_core_num: tl.constexpr):
    head_id = tl.program_id(0)
    head_num = tl.num_programs(0)
    batch_id = tl.program_id(1)
    batch_size = tl.num_programs(1)
    
    scores1_ptr = tl.make_block_ptr(
        scores1 + (batch_id * head_num + head_id) * (knn * tucker_rank),
        (knn, tucker_rank),
        (tucker_rank, 1),
        (0, 0),
        (knn_align, 16),
        (1, 0))
    
    s1 = tl.load(scores1_ptr, boundary_check=(0, 1), padding_option='zero')
    
    scores2_ptr = tl.make_block_ptr(
        scores2 + (batch_id * head_num + head_id) * (knn * tucker_rank),
        (knn, tucker_rank),
        (tucker_rank, 1),
        (0, 0),
        (knn_align, 16),
        (1, 0))
    
    s2 = tl.load(scores2_ptr, boundary_check=(0, 1), padding_option='zero')

    o_sum = tl.zeros((knn_align, knn_align), tl.float32)
    for tucker_core_id in range(0, tucker_core_num):
        tucker_core_ptr = tl.make_block_ptr(
            tucker_core + (tucker_core_id * head_num + head_id) * (tucker_rank * tucker_rank),
            (tucker_rank, tucker_rank),
            (tucker_rank, 1),
            (0, 0),
            (16, 16),
            (1, 0))

        c = tl.load(tucker_core_ptr, boundary_check=(0, 1), padding_option='zero')
        o = tl.dot(tl.dot(s1, c).to(tl.bfloat16), tl.trans(s2))

        # m = s1[:,None,:,None] * c[None,None,:,:] * s2[None,:,None,:]
        # o = tl.sum(tl.sum(m, 2), 2)
        
        score_list_ptr = tl.make_block_ptr(
            score_list + ((tucker_core_id * batch_size + batch_id) * head_num + head_id) * (knn * knn),
            (knn, knn),
            (knn, 1),
            (0, 0),
            (knn_align, knn_align),
            (1, 0))

        tl.store(score_list_ptr, o.to(tl.bfloat16), boundary_check=(0, 1))
        o_sum += o
        
    all_scores_ptr = tl.make_block_ptr(
        all_scores + (batch_id * head_num + head_id) * (knn * knn),
        (knn, knn),
        (knn, 1),
        (0, 0),
        (knn_align, knn_align),
        (1, 0))

    tl.store(all_scores_ptr, o_sum.to(tl.bfloat16), boundary_check=(0, 1))

@triton.jit
def einsum_backward(score_list_grad, scores1_grad, scores2_grad, tucker_core_grad, scores1, scores2, tucker_core, knn: tl.constexpr, knn_align: tl.constexpr, tucker_rank: tl.constexpr, tucker_core_num: tl.constexpr):
    head_id = tl.program_id(0)
    head_num = tl.num_programs(0)
    batch_id = tl.program_id(1)
    batch_size = tl.num_programs(1)
    
    scores1_ptr = tl.make_block_ptr(
        scores1 + (batch_id * head_num + head_id) * (knn * tucker_rank),
        (knn, tucker_rank),
        (tucker_rank, 1),
        (0, 0),
        (knn_align, 16),
        (1, 0))
    
    s1 = tl.load(scores1_ptr, boundary_check=(0, 1), padding_option='zero')

    scores2_ptr = tl.make_block_ptr(
        scores2 + (batch_id * head_num + head_id) * (knn * tucker_rank),
        (knn, tucker_rank),
        (tucker_rank, 1),
        (0, 0),
        (knn_align, 16),
        (1, 0))
    
    s2 = tl.load(scores2_ptr, boundary_check=(0, 1), padding_option='zero')
    
    s1_grad = tl.zeros((knn_align, 16), tl.float32)
    s2_grad = tl.zeros((knn_align, 16), tl.float32)

    for tucker_core_id in range(0, tucker_core_num):
        tucker_core_ptr = tl.make_block_ptr(
            tucker_core + (tucker_core_id * head_num + head_id) * (tucker_rank * tucker_rank),
            (tucker_rank, tucker_rank),
            (tucker_rank, 1),
            (0, 0),
            (16, 16),
            (1, 0))

        c = tl.load(tucker_core_ptr, boundary_check=(0, 1), padding_option='zero')
                
        score_list_grad_ptr = tl.make_block_ptr(
            score_list_grad + ((tucker_core_id * batch_size + batch_id) * head_num + head_id) * (knn * knn),
            (knn, knn),
            (knn, 1),
            (0, 0),
            (knn_align, knn_align),
            (1, 0))

        d_o = tl.load(score_list_grad_ptr, boundary_check=(0, 1))
        c = c.to(tl.bfloat16)
        # o = tl.dot(tl.dot(s1, c).to(tl.bfloat16), tl.trans(s2))
        s1_grad += tl.dot(d_o, tl.trans(tl.dot(c, tl.trans(s2)).to(tl.bfloat16)))
        s2_grad += tl.dot(tl.trans(d_o), tl.dot(s1, c).to(tl.bfloat16))
        c_grad = tl.dot(tl.dot(tl.trans(s1), d_o).to(tl.bfloat16), s2)
        
        tucker_core_grad_ptr = tl.make_block_ptr(
            tucker_core_grad + ((tucker_core_id * batch_size + batch_id) * head_num + head_id) * (tucker_rank * tucker_rank),
            (tucker_rank, tucker_rank),
            (tucker_rank, 1),
            (0, 0),
            (16, 16),
            (1, 0))

        tl.store(tucker_core_grad_ptr, c_grad, boundary_check=(0, 1))
        
    scores1_grad_ptr = tl.make_block_ptr(
        scores1_grad + (batch_id * head_num + head_id) * (knn * tucker_rank),
        (knn, tucker_rank),
        (tucker_rank, 1),
        (0, 0),
        (knn_align, 16),
        (1, 0))

    tl.store(scores1_grad_ptr, s1_grad.to(tl.bfloat16), boundary_check=(0, 1))
    
    scores2_grad_ptr = tl.make_block_ptr(
        scores2_grad + (batch_id * head_num + head_id) * (knn * tucker_rank),
        (knn, tucker_rank),
        (tucker_rank, 1),
        (0, 0),
        (knn_align, 16),
        (1, 0))
    
    tl.store(scores2_grad_ptr, s2_grad.to(tl.bfloat16), boundary_check=(0, 1))
 
class TritonEinsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores1, scores2, tucker_core):
        scores1 = scores1.contiguous()
        scores2 = scores2.contiguous()
        tucker_core = tucker_core.contiguous()
        ctx.save_for_backward(scores1, scores2, tucker_core)
        batch_size = scores1.shape[0]
        head_num = scores1.shape[1]
        knn = scores1.shape[2]
        knn_align = (knn + 63) // 64 * 64
        tucker_rank = scores1.shape[3]
        tucker_core_num = tucker_core.shape[0]
        score_list = torch.empty((tucker_core_num, batch_size, head_num, knn, knn), device="cuda", dtype=torch.bfloat16, requires_grad=True)
        all_scores = torch.empty((batch_size, head_num, knn, knn), device="cuda", dtype=torch.bfloat16, requires_grad=False)
        grid = (head_num, batch_size)
        einsum_forward[grid](score_list, all_scores, scores1, scores2, tucker_core, knn, knn_align, tucker_rank, tucker_core_num,
                             num_warps=knn_align//16, num_stages=2)
        return score_list, all_scores
    
    @staticmethod
    def backward(ctx, score_list_grad, all_scores_grad):
        score_list_grad = score_list_grad.contiguous()
        scores1, scores2, tucker_core = ctx.saved_tensors
        batch_size = scores1.shape[0]
        head_num = scores1.shape[1]
        knn = scores1.shape[2]
        knn_align = (knn + 63) // 64 * 64
        tucker_rank = scores1.shape[3]
        tucker_core_num = tucker_core.shape[0]
        scores1_grad = torch.empty_like(scores1)
        scores2_grad = torch.empty_like(scores2)
        tucker_core_grad = torch.empty((tucker_core_num, batch_size, head_num, tucker_rank, tucker_rank), device=tucker_core.device, dtype=torch.float32)
        grid = (head_num, batch_size)
        einsum_backward[grid](score_list_grad, scores1_grad, scores2_grad, tucker_core_grad, scores1, scores2, tucker_core, knn, knn_align, tucker_rank, tucker_core_num,
                              num_warps=4, num_stages=2)
        return scores1_grad, scores2_grad, tucker_core_grad.sum(1).to(torch.bfloat16), None, None

class FusedMulAddTopk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores_refine, tucker_core_uv, knn):
        import fused_topk

        scores_refine = scores_refine.contiguous()
        tucker_core_uv = tucker_core_uv.contiguous()

        batch_size = scores_refine.shape[1]
        head_num = scores_refine.shape[2]
        n_keys = scores_refine.shape[3]
        tucker_rank = scores_refine.shape[4]
        assert tucker_rank == 2

        scores, value, indices, count = fused_topk.fused_einsum_topk_step1_balance_loss(scores_refine, tucker_core_uv.view(2, head_num, tucker_rank).float(), knn)
        indices = indices.to(torch.int64)
        ctx.save_for_backward(tucker_core_uv, indices)
        ctx.mark_non_differentiable(indices, count)
        ctx.size = (2, batch_size, head_num, n_keys, tucker_rank)

        return scores, value, indices, count

    @staticmethod
    def backward(ctx, scores_grad, value_grad, indices_grad, count_grad):
        tucker_core_uv, indices = ctx.saved_tensors
        scores_refine_grad = scores_grad.unsqueeze(-1) * tucker_core_uv
        scores_refine_grad = scores_refine_grad + torch.zeros(ctx.size, device=value_grad.device, dtype=value_grad.dtype).scatter_(3, indices.unsqueeze(-1).repeat(1,1,1,1,2), value_grad)
        return scores_refine_grad, None, None

class FusedEinsumTopk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores1, scores2, tucker_core, final_knn):
        import fused_topk

        scores1 = scores1.contiguous()
        scores2 = scores2.contiguous()
        tucker_core = tucker_core.contiguous()
        ctx.save_for_backward(scores1, scores2, tucker_core)
        batch_size = scores1.shape[0]
        head_num = scores1.shape[1]
        knn = scores1.shape[2]
        tucker_rank = scores1.shape[3]
        assert tucker_rank == 2

        scores_chosen = torch.stack([scores1, scores2], dim=0)
        scores, score_list, best_indices = fused_topk.fused_einsum_topk_step2(scores_chosen, tucker_core.float(), final_knn)
        best_indices = best_indices.to(torch.int64).view(batch_size, head_num, -1)
        ctx.save_for_backward(scores1, scores2, tucker_core, best_indices)
        ctx.mark_non_differentiable(scores, best_indices)
        ctx.size = (2, batch_size, head_num, knn * knn)

        return scores, score_list, best_indices

    @staticmethod
    def backward(ctx, value_sum_grad, value_grad, indices_grad):
        scores1, scores2, tucker_core, indices = ctx.saved_tensors
        score_list_grad = torch.zeros(ctx.size, device=value_grad.device, dtype=value_grad.dtype).scatter_(3, indices.repeat(2, 1, 1, 1), value_grad)

        batch_size = scores1.shape[0]
        head_num = scores1.shape[1]
        knn = scores1.shape[2]
        knn_align = (knn + 63) // 64 * 64
        tucker_rank = scores1.shape[3]
        tucker_core_num = tucker_core.shape[0]
        scores1_grad = torch.empty_like(scores1)
        scores2_grad = torch.empty_like(scores2)
        tucker_core_grad = torch.empty((tucker_core_num, batch_size, head_num, tucker_rank, tucker_rank), device=tucker_core.device, dtype=torch.float32)
        score_list_grad = score_list_grad.reshape(2, batch_size, head_num, knn, knn)

        grid = (head_num, batch_size)
        einsum_backward[grid](score_list_grad, scores1_grad, scores2_grad, tucker_core_grad, scores1, scores2, tucker_core, knn, knn_align, tucker_rank, tucker_core_num,
                              num_warps=4, num_stages=2)
        return scores1_grad, scores2_grad, tucker_core_grad.sum(1).to(torch.bfloat16), None, None, None, None

class FusedLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight_bf16, main_grad, scores, padding_idx, vocab_size, per_layer_vocab_size, shift, group_size=1, has_padding_idx=True):
        # weight_bf16: bf16 view of the value table (cast at call site; same tensor if already bf16)
        # main_grad:   fp32 gradient accumulation buffer owned by the optimizer
        ctx.vocab_size, ctx.per_layer_vocab_size, ctx.shift = vocab_size, per_layer_vocab_size, shift
        ctx.padding_idx = padding_idx
        ctx.has_padding_idx = has_padding_idx
        if group_size == 0:
            group_size = 1
        ctx.group_size = group_size
        ctx.save_for_backward(indices, weight_bf16, main_grad, scores)
        import fused_lookup
        output = fused_lookup.forward(indices.contiguous(), weight_bf16.contiguous(), scores.contiguous(), vocab_size, per_layer_vocab_size, shift, group_size, padding_idx, has_padding_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, weight_bf16, main_grad, scores = ctx.saved_tensors
        import fused_lookup
        # main_grad storage is shared with the original Parameter's main_grad buffer;
        # the C++ kernel accumulates into it in-place.
        score_grad = fused_lookup.backward(indices.contiguous(), weight_bf16.contiguous(), scores.contiguous(), grad_output.contiguous(), main_grad, ctx.vocab_size, ctx.per_layer_vocab_size, ctx.shift, ctx.group_size, ctx.padding_idx, ctx.has_padding_idx)
        # inputs: indices, weight_bf16, main_grad, scores, padding_idx, vocab_size,
        #         per_layer_vocab_size, shift, group_size, has_padding_idx
        return None, None, None, score_grad, None, None, None, None, None, None


class XperfGlu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight_bf16, main_grad, p_input, vocab_size, per_layer_vocab_size, shift, group_size, padding_idx, has_padding_idx):
        # weight_bf16: bf16 view of the value table (cast at call site; same tensor if already bf16)
        # main_grad:   fp32 gradient accumulation buffer owned by the optimizer
        ctx.vocab_size, ctx.per_layer_vocab_size, ctx.shift, ctx.group_size = vocab_size, per_layer_vocab_size, shift, group_size
        ctx.save_for_backward(indices, weight_bf16, main_grad, p_input)
        import fused_glu
        output = fused_glu.forward(indices, weight_bf16, p_input, vocab_size, per_layer_vocab_size, shift, group_size, padding_idx, has_padding_idx)
        ctx.padding_idx = padding_idx
        ctx.has_padding_idx = has_padding_idx
        return output

    @staticmethod
    def backward(ctx, output_grad):
        import fused_glu
        indices, weight_bf16, main_grad, p_input = ctx.saved_tensors
        # main_grad storage is shared with the original Parameter's main_grad buffer;
        # the C++ kernel accumulates into it in-place.
        p_input_grad = fused_glu.backward(indices, weight_bf16, p_input, output_grad.contiguous(), main_grad, ctx.vocab_size, ctx.per_layer_vocab_size, ctx.shift, ctx.group_size, ctx.padding_idx, ctx.has_padding_idx)
        # inputs: indices, weight_bf16, main_grad, p_input, vocab_size,
        #         per_layer_vocab_size, shift, group_size, padding_idx, has_padding_idx
        return None, None, None, p_input_grad, None, None, None, None, None, None

