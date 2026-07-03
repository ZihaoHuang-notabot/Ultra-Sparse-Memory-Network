#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F

def pytorch_xperf_glu(
    indices, weight, p_input,
    vocab_size, per_layer_vocab_size, shift,
    group_size, padding_idx, has_padding_idx
):
    """
    OPT-1  group_id via float divide — avoids slow int64 // kernel on NPU.
    OPT-2  real_indices stays int32 — torch.where fuses pad-mask + zeroing,
           no .long() upcast, half the memory traffic on [BS, KNN].
    OPT-3  looked_up stays bf16 — cast deferred to tiny gather output ~1.6MB
           instead of looked_up [BS, KNN, D] ~600MB.
    OPT-4  einsum → bmm:
             p_input_grouped [BS, G, D] @ looked_up_t [BS, D, KNN] → [BS, G, KNN]
             permute → [BS, KNN, G] for gather.
           looked_up permute is layout-only (bf16, ~76MB), no dtype change.
    """
    bs        = indices.shape[0]
    knn       = indices.shape[1]
    embed_dim = weight.shape[1]

    # ── weight cast ───────────────────────────────────────────────────────
    weight_bf16 = weight.to(torch.bfloat16)

    # ── OPT-2: fused index arithmetic, no .long() ─────────────────────────
    is_pad       = (indices == padding_idx)
    safe_idx     = torch.where(is_pad, torch.zeros_like(indices), indices)
    real_indices = (safe_idx % per_layer_vocab_size + shift) % vocab_size   # int32

    # ── embedding → stays bf16 ────────────────────────────────────────────
    #import pdb; pdb.set_trace()

    looked_up = F.embedding(real_indices, weight_bf16)   # [BS, KNN, D]  bf16

    if group_size <= 1:
        # bmm: p_input [BS,1,D] @ looked_up_t [BS,D,KNN] → [BS,1,KNN]
        looked_up_t = looked_up.permute(0, 2, 1).contiguous()       # [BS, D, KNN]  bf16
        scores      = torch.bmm(
            p_input.unsqueeze(1), looked_up_t
        ).squeeze(1).float()                                         # [BS, KNN]  fp32
    else:
        # ── OPT-1: group_id via float divide ──────────────────────────────
        group_id = (safe_idx.float() / per_layer_vocab_size).to(torch.int32)  # [BS, KNN]

        # OPT-5: flip BMM — transpose p_input_grouped (~1.6MB) not looked_up (~76MB)
        # optimized
        p_input_grouped  = p_input.view(bs, group_size, embed_dim)       # [BS, G, D]  bf16
        all_group_scores = torch.einsum('bgd,bkd->bkg',
                                        p_input_grouped, looked_up)       # [BS, KNN, G]  bf16
        scores = all_group_scores.gather(
            2, group_id.unsqueeze(-1)).squeeze(-1).float()                # [BS, KNN]  fp32

    # ── padding mask (reuse is_pad — no extra kernel) ─────────────────────
    if has_padding_idx:
        scores = scores.masked_fill(is_pad, 0.0)

    return scores.to(torch.bfloat16)





#pytorch_fused_lookup_optimized
def pytorch_fused_lookup(
    indices, weight, scores, padding_idx,
    vocab_size, per_layer_vocab_size, shift,
    group_size, has_padding_idx
):
    """
    OPT-1  No .long() upcast; torch.where fuses pad-mask + index zeroing.
           All index arithmetic stays int32.

    OPT-2  values stays bf16 after embedding — NO cast at all.
           Eliminates the 3ms bf16→fp32 cast of [BS, KNN, D].

    OPT-3  score_matrix built directly in bf16 (matches values dtype for bmm).

    OPT-4  Flipped BMM: score_matrix_t[BS,8,KNN] @ values[BS,KNN,D] → [BS,8,D]
           Contiguous copy on score_matrix (~50MB) not values (~2.4GB).

    OPT-5  Cast happens on the SMALL result [BS,8,D] ~12MB, not on inputs.
           output accumulation in fp32 as before.
    """
    
    CUDA_WMMA_SCORE_COLS = 8
    WMMA_KNN_TILE        = 16

    bs        = indices.shape[0]
    knn       = indices.shape[1]
    embed_dim = weight.shape[1]

    tucker_core_num = scores.shape[0] if scores.dim() == 3 else 1
    each_core_dim   = embed_dim // tucker_core_num

    # ── weight cast ───────────────────────────────────────────────────────
    weight_bf16 = weight.to(torch.bfloat16)

    # ── OPT-1: fused index arithmetic, no .long() ─────────────────────────
    is_pad       = (indices == padding_idx)
    safe_idx     = torch.where(is_pad, torch.zeros_like(indices), indices)
    real_indices = (safe_idx % per_layer_vocab_size + shift) % vocab_size
    group_id     = (safe_idx.float() / per_layer_vocab_size).to(torch.int32)
    padding_mask = is_pad
    #import pdb; pdb.set_trace()
    # ── embedding → values stays bf16, no cast ────────────────────────────
    values = F.embedding(real_indices, weight_bf16)   # [BS, KNN, D]  bf16

    # ── knn alignment padding ─────────────────────────────────────────────
    knn_align = (knn + WMMA_KNN_TILE - 1) // WMMA_KNN_TILE * WMMA_KNN_TILE
    if knn_align > knn:
        pad          = knn_align - knn
        values       = F.pad(values,       (0, 0, 0, pad))
        group_id     = F.pad(group_id,     (0, pad))
        padding_mask = F.pad(padding_mask, (0, pad), value=True)

    scores_3d = scores.unsqueeze(0) if scores.dim() == 2 else scores
    output    = torch.zeros(bs, group_size, embed_dim, device=values.device, dtype=torch.float32)

    # pre-compute gid once (shared across heads)
    gid = group_id[:, :knn].clamp(0, CUDA_WMMA_SCORE_COLS - 1)
    if knn_align > knn:
        gid = F.pad(gid, (0, knn_align - knn))

    for head in range(tucker_core_num):
        col_start, col_end = head * each_core_dim, (head + 1) * each_core_dim
        s = scores_3d[head]   # [BS, KNN]  bf16

        # OPT-3: score_matrix in bf16 directly
        score_matrix = torch.zeros(bs, knn_align, CUDA_WMMA_SCORE_COLS,
                                   device=values.device, dtype=torch.bfloat16)
        src_bf16 = (s.unsqueeze(-1) if knn_align == knn
                    else F.pad(s, (0, knn_align - knn)).unsqueeze(-1))
        score_matrix.scatter_(2, gid.unsqueeze(-1), src_bf16)
        if has_padding_idx:
            score_matrix = score_matrix.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # OPT-4 + OPT-5: flip BMM, cast only the small result
        # score_matrix_t : [BS, 8, KNN_ALIGN]  bf16  ~50MB permute
        # values_head    : [BS, KNN_ALIGN, D_head]  bf16  free slice
        # result         : [BS, 8, D_head]  bf16  → cast to fp32 ~12MB
        score_matrix_t = score_matrix.permute(0, 2, 1).contiguous()  # [BS, 8, KNN_ALIGN]
        values_head    = values[:, :, col_start:col_end]              # [BS, KNN_ALIGN, D_head]
        result         = torch.bmm(score_matrix_t, values_head)       # [BS, 8, D_head]  bf16
        output[:, :, col_start:col_end] += result[:, :group_size, :].float()

    return output.view(bs, group_size * embed_dim).to(torch.bfloat16)

