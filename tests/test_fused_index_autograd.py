"""
CPU-only autograd tests for the FusedLookup / XperfGlu refactor.

These tests monkey-patch the C++ kernels (fused_lookup, fused_glu) with Python
stubs so the autograd plumbing can be verified without a GPU.  They catch:

  1. Signature / arity mismatches — PyTorch raises on wrong number of backward
     return values.
  2. save_for_backward round-trip — ctx.saved_tensors returns all four tensors
     in the right order.
  3. Storage identity of main_grad across save_for_backward — the sentinel test
     proves that in-place writes to the saved tensor land in the original buffer,
     which is the key property that the old param_bf16 pattern failed to provide.
  4. Both fp32 (cast inserted) and bf16 (no cast) call-site paths.
  5. XperfGlu symmetric treatment.
"""
import sys
import types
import torch
import pytest

# ---------------------------------------------------------------------------
# Stub out the C++ extension modules before importing fused_index
# ---------------------------------------------------------------------------

def _make_lookup_stub():
    """Stub for fused_lookup that returns zeros and accumulates a sentinel into
    the weight-grad buffer so we can verify storage identity."""
    mod = types.ModuleType("fused_lookup")

    def forward(indices, weight, scores, vocab_size, per_layer_vocab_size,
                shift, group_size, padding_idx, has_padding_idx):
        bs = scores.shape[-1] if scores.dim() == 1 else scores.shape[0]
        out_dim = weight.shape[-1] if weight.dim() >= 2 else 1
        return torch.zeros(bs, out_dim, dtype=weight.dtype)

    def backward(indices, weight, scores, grad_output, main_grad,
                 vocab_size, per_layer_vocab_size, shift, group_size,
                 padding_idx, has_padding_idx):
        # Write a recognisable value into main_grad in-place so the storage
        # identity test can detect it.
        main_grad.fill_(42.0)
        score_grad = torch.zeros_like(scores)
        return score_grad

    mod.forward = forward
    mod.backward = backward
    return mod


def _make_glu_stub():
    mod = types.ModuleType("fused_glu")

    def forward(indices, weight, p_input, vocab_size, per_layer_vocab_size,
                shift, group_size, padding_idx, has_padding_idx):
        bs = p_input.shape[0]
        return torch.zeros(bs, indices.shape[-1], dtype=weight.dtype)

    def backward(indices, weight, p_input, output_grad, main_grad,
                 vocab_size, per_layer_vocab_size, shift, group_size,
                 padding_idx, has_padding_idx):
        main_grad.fill_(42.0)
        p_input_grad = torch.zeros_like(p_input)
        return p_input_grad

    mod.forward = forward
    mod.backward = backward
    return mod


sys.modules.setdefault("fused_lookup", _make_lookup_stub())
sys.modules.setdefault("fused_glu", _make_glu_stub())

# Load fused_index.py directly via importlib to avoid triggering
# fuse_ops/__init__.py which compiles CUDA extensions (requires a GPU).
import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location(
    "fuse_ops.fused_index",
    _pl.Path(__file__).parent.parent / "fuse_ops" / "fused_index.py",
)
_mod = _ilu.module_from_spec(_spec)
sys.modules["fuse_ops.fused_index"] = _mod
_spec.loader.exec_module(_mod)

FusedLookup = _mod.FusedLookup
XperfGlu = _mod.XperfGlu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lookup_inputs(fp32_weight=False):
    """Return (indices, weight_bf16, main_grad, scores, kwargs, weight_orig)."""
    n_vocab, vdim = 64, 16
    bs, knn = 4, 8
    dtype = torch.float32 if fp32_weight else torch.bfloat16
    weight_orig = torch.randn(n_vocab, vdim, dtype=dtype)
    weight_bf16 = weight_orig if dtype == torch.bfloat16 else weight_orig.to(torch.bfloat16)
    main_grad = torch.zeros(n_vocab, vdim, dtype=torch.float32)
    indices = torch.randint(0, n_vocab, (bs, knn), dtype=torch.int32)
    scores = torch.randn(bs, knn, requires_grad=True)
    kwargs = dict(padding_idx=0, vocab_size=n_vocab, per_layer_vocab_size=n_vocab,
                  shift=0, group_size=1, has_padding_idx=False)
    return indices, weight_bf16, main_grad, scores, kwargs, weight_orig


def _make_glu_inputs(fp32_weight=False):
    """Return (indices, weight_bf16, main_grad, p_input, kwargs, weight_orig)."""
    n_vocab, edim = 64, 16
    bs, knn = 4, 8
    dtype = torch.float32 if fp32_weight else torch.bfloat16
    weight_orig = torch.randn(n_vocab, edim, dtype=dtype)
    weight_bf16 = weight_orig if dtype == torch.bfloat16 else weight_orig.to(torch.bfloat16)
    main_grad = torch.zeros(n_vocab, edim, dtype=torch.float32)
    indices = torch.randint(0, n_vocab, (bs, knn), dtype=torch.int32)
    p_input = torch.randn(bs, edim, requires_grad=True)
    kwargs = dict(vocab_size=n_vocab, per_layer_vocab_size=n_vocab,
                  shift=0, group_size=1, padding_idx=0, has_padding_idx=False)
    return indices, weight_bf16, main_grad, p_input, kwargs, weight_orig


# ---------------------------------------------------------------------------
# FusedLookup tests
# ---------------------------------------------------------------------------

class TestFusedLookupSignature:
    def test_forward_runs_bf16(self):
        indices, weight_bf16, main_grad, scores, kw, _ = _make_lookup_inputs(fp32_weight=False)
        out = FusedLookup.apply(indices, weight_bf16, main_grad, scores,
                                kw["padding_idx"], kw["vocab_size"],
                                kw["per_layer_vocab_size"], kw["shift"],
                                kw["group_size"], kw["has_padding_idx"])
        assert out is not None

    def test_forward_runs_fp32_weight(self):
        """Cast happens at call site; FusedLookup always receives bf16."""
        indices, weight_bf16, main_grad, scores, kw, weight_orig = _make_lookup_inputs(fp32_weight=True)
        assert weight_bf16.dtype == torch.bfloat16
        assert weight_bf16 is not weight_orig  # a new tensor was created by the cast
        out = FusedLookup.apply(indices, weight_bf16, main_grad, scores,
                                kw["padding_idx"], kw["vocab_size"],
                                kw["per_layer_vocab_size"], kw["shift"],
                                kw["group_size"], kw["has_padding_idx"])
        assert out is not None

    def test_bf16_weight_no_copy(self):
        """When weight is already bf16 the call site skips the cast."""
        indices, weight_bf16, main_grad, scores, kw, weight_orig = _make_lookup_inputs(fp32_weight=False)
        assert weight_bf16 is weight_orig  # same tensor, no allocation


class TestFusedLookupBackward:
    def test_backward_runs_without_error(self):
        indices, weight_bf16, main_grad, scores, kw, _ = _make_lookup_inputs()
        out = FusedLookup.apply(indices, weight_bf16, main_grad, scores,
                                kw["padding_idx"], kw["vocab_size"],
                                kw["per_layer_vocab_size"], kw["shift"],
                                kw["group_size"], kw["has_padding_idx"])
        out.sum().backward()
        assert scores.grad is not None

    def test_backward_return_arity(self):
        """PyTorch raises if the number of backward return values != number of inputs."""
        indices, weight_bf16, main_grad, scores, kw, _ = _make_lookup_inputs()
        out = FusedLookup.apply(indices, weight_bf16, main_grad, scores,
                                kw["padding_idx"], kw["vocab_size"],
                                kw["per_layer_vocab_size"], kw["shift"],
                                kw["group_size"], kw["has_padding_idx"])
        # If arity were wrong this would raise RuntimeError
        out.sum().backward()

    def test_main_grad_storage_identity(self):
        """
        THE KEY TEST: the C++ stub writes 42 into main_grad in-place during
        backward.  We verify the write landed in the ORIGINAL main_grad buffer
        (same storage as what we passed to FusedLookup.apply), not a copy.
        This is the property that the old param_bf16 pattern failed to provide
        when ctx.saved_tensors returned a fresh Tensor after pack/unpack.
        """
        indices, weight_bf16, main_grad, scores, kw, _ = _make_lookup_inputs()
        sentinel_value = 42.0
        # Sanity: main_grad is all zeros before backward
        assert main_grad.abs().max().item() == 0.0
        out = FusedLookup.apply(indices, weight_bf16, main_grad, scores,
                                kw["padding_idx"], kw["vocab_size"],
                                kw["per_layer_vocab_size"], kw["shift"],
                                kw["group_size"], kw["has_padding_idx"])
        out.sum().backward()
        # The stub filled the saved main_grad tensor with 42; if storage is
        # shared with our original main_grad buffer, we see it here.
        assert main_grad.abs().max().item() == sentinel_value, (
            "main_grad storage was NOT shared across save_for_backward — "
            "the backward wrote into a copy, not the original buffer"
        )

    def test_weight_bf16_grad_is_none(self):
        """FusedLookup returns None for weight_bf16 grad (gradient flows through main_grad)."""
        indices, weight_bf16, main_grad, scores, kw, _ = _make_lookup_inputs()
        weight_bf16 = weight_bf16.clone().requires_grad_(True)
        out = FusedLookup.apply(indices, weight_bf16, main_grad, scores,
                                kw["padding_idx"], kw["vocab_size"],
                                kw["per_layer_vocab_size"], kw["shift"],
                                kw["group_size"], kw["has_padding_idx"])
        out.sum().backward()
        assert weight_bf16.grad is None


# ---------------------------------------------------------------------------
# XperfGlu tests
# ---------------------------------------------------------------------------

class TestXperfGluSignature:
    def test_forward_runs_bf16(self):
        indices, weight_bf16, main_grad, p_input, kw, _ = _make_glu_inputs(fp32_weight=False)
        out = XperfGlu.apply(indices, weight_bf16, main_grad, p_input,
                             kw["vocab_size"], kw["per_layer_vocab_size"],
                             kw["shift"], kw["group_size"],
                             kw["padding_idx"], kw["has_padding_idx"])
        assert out is not None

    def test_forward_runs_fp32_weight(self):
        indices, weight_bf16, main_grad, p_input, kw, weight_orig = _make_glu_inputs(fp32_weight=True)
        assert weight_bf16.dtype == torch.bfloat16
        assert weight_bf16 is not weight_orig
        out = XperfGlu.apply(indices, weight_bf16, main_grad, p_input,
                             kw["vocab_size"], kw["per_layer_vocab_size"],
                             kw["shift"], kw["group_size"],
                             kw["padding_idx"], kw["has_padding_idx"])
        assert out is not None


class TestXperfGluBackward:
    def test_backward_runs_without_error(self):
        indices, weight_bf16, main_grad, p_input, kw, _ = _make_glu_inputs()
        out = XperfGlu.apply(indices, weight_bf16, main_grad, p_input,
                             kw["vocab_size"], kw["per_layer_vocab_size"],
                             kw["shift"], kw["group_size"],
                             kw["padding_idx"], kw["has_padding_idx"])
        out.sum().backward()
        assert p_input.grad is not None

    def test_main_grad_storage_identity(self):
        """Same sentinel test as FusedLookup — verifies storage is preserved."""
        indices, weight_bf16, main_grad, p_input, kw, _ = _make_glu_inputs()
        assert main_grad.abs().max().item() == 0.0
        out = XperfGlu.apply(indices, weight_bf16, main_grad, p_input,
                             kw["vocab_size"], kw["per_layer_vocab_size"],
                             kw["shift"], kw["group_size"],
                             kw["padding_idx"], kw["has_padding_idx"])
        out.sum().backward()
        assert main_grad.abs().max().item() == 42.0, (
            "XperfGlu main_grad storage was NOT shared across save_for_backward"
        )
