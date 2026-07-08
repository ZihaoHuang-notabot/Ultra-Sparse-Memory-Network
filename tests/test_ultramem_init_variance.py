"""
CPU-only sanity check for UltraMemV2 init scaling.

Catches the two failure modes that broke prior runs even with Eq. 26 in place:
  1. `_measure_sigma_s_squared` returning NaN / non-finite / wildly off scale.
  2. `values_proj` not depth-scaled (would put memory residual ~sqrt(2L)x
     larger than FFN residual at init, regardless of sigma_V correctness).

These are config-level checks — no GPU and no FSDP needed.
"""
import math

import pytest
import torch

from olmo.config import InitFnType, ModelConfig
from olmo.ultramem_layer_v2 import UltraMemLayerV2


def _make_config(n_layers=8, init_fn=InitFnType.full_megatron, init_std=0.02282):
    cfg = ModelConfig()
    cfg.n_layers = n_layers
    cfg.d_model = 512
    cfg.init_fn = init_fn
    cfg.init_std = init_std
    cfg.init_device = "cpu"
    cfg.mlp_ratio = 8
    cfg.mem_q_for_each_tucker_rank = True
    cfg.mem_use_glu_act = False
    cfg.mem_key_expand_time = 2
    cfg.mem_output_dropout_rate = 0.0
    cfg.vertical_parallel = False
    cfg.mem_parallel_size = 1
    return cfg


def _make_layer(has_value, n_layers=8, init_fn=InitFnType.full_megatron):
    cfg = _make_config(n_layers=n_layers, init_fn=init_fn)
    return UltraMemLayerV2(
        hidden_size=512, knum=64, kdim=128, vdim=144, pre_vdim=48,
        knn=32, head=1, tucker_rank=2, tucker_multihead=2,
        value_expand_time=1, tucker_rank_penalty=0.0,
        layer_id=0, has_value=has_value,
        share_ratio=1.0, mem_layer_num=n_layers, config=cfg,
    )


def test_sigma_s_squared_is_finite_and_small():
    """sigma_s^2 should be a finite positive number well under 10 — paper hint
    puts it near 0.1.  Anything larger means the calibration is mis-scaled and
    Eq. 26 will produce a meaningless sigma_V."""
    layer = _make_layer(has_value=False)
    sigma_s_sq = layer._measure_sigma_s_squared(n_samples=512)
    assert math.isfinite(sigma_s_sq)
    assert 0.0 < sigma_s_sq < 10.0, f"sigma_s^2 out of expected range: {sigma_s_sq}"


def test_values_proj_std_depth_scales_under_full_megatron():
    """Paper Eq. 26 derivation assumes ff_out is 1/sqrt(2L)-scaled.  For
    init_fn=full_megatron, values_proj must follow the same scaling so the
    memory residual matches the FFN residual at init."""
    layer = _make_layer(has_value=False, n_layers=20, init_fn=InitFnType.full_megatron)
    expected = layer.config.init_std / math.sqrt(2.0 * 20)
    got = layer._compute_values_proj_std()
    assert abs(got - expected) < 1e-9, (got, expected)


def test_values_proj_std_falls_back_under_normal():
    """For init_fn=normal (no depth scaling), values_proj should match
    init_std so the memory residual matches the FFN residual."""
    layer = _make_layer(has_value=False, init_fn=InitFnType.normal)
    got = layer._compute_values_proj_std()
    assert abs(got - 0.02282) < 1e-9, got


def test_eq26_value_init_runs_and_produces_reasonable_std():
    """End-to-end: reset_parameters on a has_value=True layer should populate
    values_for_look_up with finite, non-zero std reasonably close to sigma_V."""
    layer = _make_layer(has_value=True)
    layer.reset_parameters(distributed_strategy=None)
    assert layer.values_for_look_up.isfinite().all()
    actual_std = layer.values_for_look_up.float().std().item()
    # sigma_V in this config is tiny — assert it's finite, positive, and < 1.
    assert 0.0 < actual_std < 1.0, actual_std
