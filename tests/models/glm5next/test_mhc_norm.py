# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.layers.mhc as mhc_layers
from vllm.model_executor.kernels.mhc import mhc_post_torch, mhc_pre_torch
from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, MHCPreOp

NUM_TOKENS = 2
HC_MULT = 4
HIDDEN_SIZE = 32
RMS_EPS = 1e-5
HC_PRE_EPS = 1e-3
HC_SINKHORN_EPS = 1e-3
HC_POST_MULT = 1.5
SINKHORN_REPEAT = 3
NORM_EPS = 1e-3


def _rms_norm_oracle(
    layer_input: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor:
    if norm_weight is None:
        return layer_input
    x = layer_input.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm_eps)
    x = x * norm_weight.float()
    return x.to(layer_input.dtype)


def _mhc_inputs(norm_weight_dtype: torch.dtype | None):
    torch.manual_seed(0)
    hc_mult3 = 2 * HC_MULT + HC_MULT * HC_MULT
    residual = torch.randn(NUM_TOKENS, HC_MULT, HIDDEN_SIZE).bfloat16()
    fn = torch.randn(hc_mult3, HC_MULT * HIDDEN_SIZE) * 0.01
    hc_scale = torch.tensor([0.8, 1.1, 0.9], dtype=torch.float32)
    hc_base = torch.linspace(-0.2, 0.2, hc_mult3, dtype=torch.float32)
    norm_weight = None
    if norm_weight_dtype is not None:
        norm_weight = torch.linspace(
            0.5, 1.5, HIDDEN_SIZE, dtype=torch.float32
        ).to(norm_weight_dtype)
    return residual, fn, hc_scale, hc_base, norm_weight


def _run_mhc_pre(residual, fn, hc_scale, hc_base, norm_weight):
    return object.__new__(MHCPreOp).forward_native(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
    )


def _force_mhc_fallback(monkeypatch):
    monkeypatch.setattr(mhc_layers, "HAS_AITER_MHC", False)
    monkeypatch.setattr(mhc_layers, "HAS_TILELANG_MHC", False)


def _record_norm_calls(monkeypatch):
    calls = []
    apply_norm = mhc_layers._apply_mhc_norm

    def record_apply_norm(layer_input, norm_weight, norm_eps):
        calls.append(1)
        return apply_norm(layer_input, norm_weight, norm_eps)

    monkeypatch.setattr(mhc_layers, "_apply_mhc_norm", record_apply_norm)
    return calls


@pytest.mark.parametrize(
    "norm_weight_dtype",
    [torch.float32, torch.bfloat16, None],
    ids=["fp32", "bf16", "none"],
)
def test_mhc_pre_native_applies_norm_only_to_layer_input(norm_weight_dtype):
    residual, fn, hc_scale, hc_base, norm_weight = _mhc_inputs(norm_weight_dtype)
    expected = mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )

    actual = _run_mhc_pre(residual, fn, hc_scale, hc_base, norm_weight)

    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    expected_layer_input = _rms_norm_oracle(expected[2], norm_weight, NORM_EPS)
    tol = 1e-2 if norm_weight_dtype == torch.bfloat16 else 0
    torch.testing.assert_close(actual[2], expected_layer_input, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "norm_weight_dtype",
    [torch.float32, torch.bfloat16, None],
    ids=["fp32", "bf16", "none"],
)
def test_mhc_pre_hip_fallback_matches_native_and_applies_norm_once(
    monkeypatch, norm_weight_dtype
):
    _force_mhc_fallback(monkeypatch)
    calls = _record_norm_calls(monkeypatch)
    inputs = _mhc_inputs(norm_weight_dtype)

    native = _run_mhc_pre(*inputs)
    assert len(calls) == 1
    calls.clear()
    fallback = object.__new__(MHCPreOp).forward_hip(
        *inputs[:4],
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=inputs[4],
        norm_eps=NORM_EPS,
    )

    assert len(calls) == 1
    for actual, expected in zip(fallback, native, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "norm_weight_dtype",
    [torch.float32, torch.bfloat16, None],
    ids=["fp32", "bf16", "none"],
)
def test_mhc_fused_native_equals_post_then_pre_then_norm(norm_weight_dtype):
    residual, fn, hc_scale, hc_base, norm_weight = _mhc_inputs(norm_weight_dtype)
    torch.manual_seed(1)
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE).bfloat16()
    post_layer_mix = torch.randn(NUM_TOKENS, HC_MULT, 1)
    comb_res_mix = torch.softmax(torch.randn(NUM_TOKENS, HC_MULT, HC_MULT), dim=-1)

    post_expected = mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    pre_expected = mhc_pre_torch(
        post_expected,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )
    actual = object.__new__(MHCFusedPostPreOp).forward_native(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
    )

    torch.testing.assert_close(actual[0], post_expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[1], pre_expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], pre_expected[1], atol=0, rtol=0)
    expected_layer_input = _rms_norm_oracle(pre_expected[2], norm_weight, NORM_EPS)
    tol = 1e-2 if norm_weight_dtype == torch.bfloat16 else 0
    torch.testing.assert_close(actual[3], expected_layer_input, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "norm_weight_dtype",
    [torch.float32, torch.bfloat16, None],
    ids=["fp32", "bf16", "none"],
)
def test_mhc_fused_hip_fallback_matches_native_and_applies_norm_once(
    monkeypatch, norm_weight_dtype
):
    _force_mhc_fallback(monkeypatch)
    calls = _record_norm_calls(monkeypatch)
    residual, fn, hc_scale, hc_base, norm_weight = _mhc_inputs(norm_weight_dtype)
    torch.manual_seed(1)
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE).bfloat16()
    post_layer_mix = torch.randn(NUM_TOKENS, HC_MULT, 1)
    comb_res_mix = torch.softmax(torch.randn(NUM_TOKENS, HC_MULT, HC_MULT), dim=-1)
    inputs = (x, residual, post_layer_mix, comb_res_mix, fn)

    native = object.__new__(MHCFusedPostPreOp).forward_native(
        *inputs,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
    )
    assert len(calls) == 1
    calls.clear()
    fallback = object.__new__(MHCFusedPostPreOp).forward_hip(
        *inputs,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
    )

    assert len(calls) == 1
    for actual, expected in zip(fallback, native, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
