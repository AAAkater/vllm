# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm


class _TupleLinear(nn.Linear):
    def forward(self, x: torch.Tensor):
        return (super().forward(x),)


class _RecordingMLAAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shapes: tuple[torch.Size, torch.Size, torch.Size] | None = None

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del output_shape, q_dcp_replicated
        self.shapes = (q.shape, kv_c_normed.shape, k_pe.shape)
        return torch.zeros(q.shape[0], 1, dtype=q.dtype, device=q.device)


def _rmsnorm_oracle(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_fp32 = x.float()
    norm = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return (norm * weight.float()).to(x.dtype)


def test_fused_q_kv_rmsnorm_cpu_matches_independent_oracle() -> None:
    torch.manual_seed(0)
    qr = torch.randn(7, 1536)
    kv = torch.randn(7, 576)
    q_weight = torch.randn(1536)
    kv_weight = torch.randn(576)

    qr_out, kv_out = fused_q_kv_rmsnorm(
        qr, kv, q_weight, kv_weight, eps=1e-5
    )

    torch.testing.assert_close(qr_out, _rmsnorm_oracle(qr, q_weight, 1e-5))
    torch.testing.assert_close(kv_out, _rmsnorm_oracle(kv, kv_weight, 1e-5))
    assert qr_out.is_contiguous() and kv_out.is_contiguous()


def test_nope_mla_keeps_zero_width_rope_axis() -> None:
    wrapper = object.__new__(MultiHeadLatentAttentionWrapper)
    nn.Module.__init__(wrapper)
    wrapper.q_lora_rank = 4
    wrapper.kv_lora_rank = 3
    wrapper.qk_nope_head_dim = 2
    wrapper.qk_rope_head_dim = 0
    wrapper.qk_head_dim = 2
    wrapper.v_head_dim = 1
    wrapper.num_heads = 1
    wrapper.fused_qkv_a_proj = _TupleLinear(2, 7)
    wrapper.kv_a_proj_with_mqa = None
    wrapper.q_a_layernorm = nn.LayerNorm(4)
    wrapper.q_b_proj = _TupleLinear(4, 2)
    wrapper.q_proj = None
    wrapper.kv_a_layernorm = nn.LayerNorm(3)
    wrapper.o_proj = _TupleLinear(1, 2)
    wrapper.rotary_emb = None
    wrapper.indexer = None
    wrapper.indexer_rope_emb = None
    wrapper.is_sparse = False
    wrapper.skip_topk = False
    wrapper.fuse_qkv_rmsnorm = False
    wrapper.g_proj = None
    wrapper.dcp_q_replicate = False
    wrapper.mla_attn = _RecordingMLAAttention()

    hidden_states = torch.randn(2, 2)
    output = wrapper.forward(torch.arange(2), hidden_states)

    assert wrapper.mla_attn.shapes == (
        torch.Size([2, 1, 2]),
        torch.Size([2, 3]),
        torch.Size([2, 1, 0]),
    )
    assert output.shape == hidden_states.shape


def test_glm5next_modules_import_without_hpu() -> None:
    from vllm.models.glm5next.nvidia import attention, kda, model

    assert attention.Glm5NextMLAAttention.__name__ == "Glm5NextMLAAttention"
    assert kda.Glm5NextLinearAttention.__name__ == "Glm5NextLinearAttention"
    assert model.Glm5NextModel.__name__ == "Glm5NextModel"


def test_hpu_4d_bucket_inputs_flatten_to_token_major() -> None:
    from vllm.models.glm5next.nvidia.model import _flatten_bucket_inputs

    hidden_states = torch.empty(2, 1, 3, 8)
    positions = torch.empty(2, 1, 3, dtype=torch.long)
    flat_hidden, flat_positions = _flatten_bucket_inputs(
        hidden_states, positions
    )

    assert flat_hidden.shape == (6, 8)
    assert flat_positions.shape == (6,)


def test_nope_rotary_guard_skips_zero_width_rope() -> None:
    from vllm.models.glm5next.nvidia.attention import _skip_rotary_embedding

    assert _skip_rotary_embedding(0, skip_rope=False)
    assert _skip_rotary_embedding(64, skip_rope=True)
    assert not _skip_rotary_embedding(64, skip_rope=False)


def test_non_cuda_kda_native_kernels_fail_loudly() -> None:
    from vllm.models.glm5next.nvidia import kda

    if kda.NATIVE_KDA_KERNELS_AVAILABLE:
        pytest.skip("CUDA/ROCm native KDA kernels are available")

    with pytest.raises(RuntimeError, match="must replace Glm5NextLinearAttention"):
        kda.chunk_kda_with_fused_gate()
    with pytest.raises(RuntimeError, match="must replace Glm5NextLinearAttention"):
        kda.fused_recurrent_kda()


def test_language_model_only_mapper_drops_only_visual_weights() -> None:
    from vllm.models.glm5next.nvidia.model import _weights_mapper_for_mode

    names = [
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "visual.merger.weight",
    ]
    assert _weights_mapper_for_mode(False).apply_list(names) == [
        "language_model.lm_head.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "visual.blocks.0.attn.qkv.weight",
        "visual.merger.weight",
    ]
    assert _weights_mapper_for_mode(True).apply_list(names) == [
        "language_model.lm_head.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
    ]


def test_dense_hpu_indexer_weights_are_skipped_only_when_unmodeled() -> None:
    from vllm.models.glm5next.nvidia.model import (
        _is_unmodeled_dense_indexer_weight,
    )

    name = "model.layers.3.self_attn.indexer.wk.weight"
    assert _is_unmodeled_dense_indexer_weight(
        name, {}, is_hpu=True, sparse_enabled=False
    )
    assert not _is_unmodeled_dense_indexer_weight(
        name, {}, is_hpu=False, sparse_enabled=False
    )
    modeled = {
        "model.layers.3.self_attn.indexer.wk.weight": nn.Parameter(
            torch.empty(1)
        )
    }
    assert not _is_unmodeled_dense_indexer_weight(
        name, modeled, is_hpu=True, sparse_enabled=False
    )


def test_unknown_checkpoint_weight_is_rejected() -> None:
    from vllm.models.glm5next.nvidia.model import _get_checkpoint_param

    param = nn.Parameter(torch.empty(1))
    assert _get_checkpoint_param("known", {"known": param}) is param
    with pytest.raises(ValueError, match="has no matching model parameter"):
        _get_checkpoint_param("unknown", {"known": param})


def test_hpu_runtime_guard_fails_closed_for_dense_diagnostics() -> None:
    from vllm.models.glm5next.nvidia.model import _validate_glm53_hpu_runtime

    def runtime_config(
        *,
        max_model_len: int = 2048,
        prefix_caching: bool = False,
        speculative_config=None,
        pipeline_parallel_size: int = 1,
    ):
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(index_topk=2048),
                max_model_len=max_model_len,
            ),
            cache_config=SimpleNamespace(
                enable_prefix_caching=prefix_caching
            ),
            speculative_config=speculative_config,
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=pipeline_parallel_size
            ),
        )

    _validate_glm53_hpu_runtime(runtime_config(), device_type="hpu")
    _validate_glm53_hpu_runtime(
        runtime_config(
            max_model_len=1_000_000,
            prefix_caching=True,
            speculative_config=object(),
            pipeline_parallel_size=2,
        ),
        device_type="cuda",
    )

    with pytest.raises(ValueError, match="max_model_len <= 2048"):
        _validate_glm53_hpu_runtime(
            runtime_config(max_model_len=2049), device_type="hpu"
        )
    with pytest.raises(ValueError, match="no-enable-prefix-caching"):
        _validate_glm53_hpu_runtime(
            runtime_config(prefix_caching=True), device_type="hpu"
        )
    with pytest.raises(ValueError, match="speculative decoding or MTP"):
        _validate_glm53_hpu_runtime(
            runtime_config(speculative_config=object()), device_type="hpu"
        )
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        _validate_glm53_hpu_runtime(
            runtime_config(pipeline_parallel_size=2), device_type="hpu"
        )


def test_fp8_dequant_requires_cpu_serialized_source() -> None:
    from vllm.models.glm5next.nvidia.model import (
        _dequant_fp8_block,
        _require_cpu_fp8_source,
    )

    cpu_weight = torch.ones((2, 2), dtype=torch.float8_e4m3fn)
    cpu_scale = torch.ones((1, 1), dtype=torch.float32)
    _require_cpu_fp8_source(
        cpu_weight, "indexer.wk.weight", target_device_type="hpu"
    )
    dequantized = _dequant_fp8_block(cpu_weight, cpu_scale, block_size=2)
    assert dequantized.dtype == torch.bfloat16
    torch.testing.assert_close(
        dequantized.float(), torch.ones((2, 2), dtype=torch.float32)
    )

    hpu_like_weight = torch.empty(
        (2, 2), dtype=torch.float8_e4m3fn, device="meta"
    )
    with pytest.raises(RuntimeError, match="must be dequantized on CPU"):
        _require_cpu_fp8_source(
            hpu_like_weight, "indexer.wk.weight", target_device_type="hpu"
        )
    _require_cpu_fp8_source(
        hpu_like_weight, "indexer.wk.weight", target_device_type="cuda"
    )


def test_mtp_weight_range_is_explicit() -> None:
    from vllm.models.glm5next.nvidia.model import (
        get_spec_layer_idx_from_weight_name,
    )

    config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    assert get_spec_layer_idx_from_weight_name(
        config, "model.layers.1.mlp.down_proj.weight"
    ) is None
    assert get_spec_layer_idx_from_weight_name(
        config, "model.layers.2.mlp.down_proj.weight"
    ) == 2
    assert get_spec_layer_idx_from_weight_name(
        config, "model.layers.3.mlp.down_proj.weight"
    ) is None
