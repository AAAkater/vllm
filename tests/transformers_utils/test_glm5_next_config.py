# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest

from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
)


def _text_config(**overrides):
    dense_layers = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43}
    layer_types = [
        "deepseek_sparse_attention" if i in dense_layers else "linear_attention"
        for i in range(45)
    ]
    config = {
        "num_hidden_layers": 45,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        "first_k_dense_replace": 3,
        "num_experts_per_tok": 8,
        "num_experts_per_token": None,
        "hc_mult": 4,
        "mla_use_nope": True,
        "linear_attn_config": {
            "num_heads": 64,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    }
    config.update(overrides)
    return config


def test_glm5_next_preserves_checkpoint_layer_semantics():
    config = Glm5NextConfig(
        text_config=_text_config(),
        vision_config={"rms_norm_eps": 1e-5},
    )
    text_config = config.text_config

    assert len(text_config.layer_types) == 45
    assert Counter(text_config.layer_types) == {
        "linear_attention": 34,
        "deepseek_sparse_attention": 11,
    }
    assert text_config.num_experts_per_token == 8
    assert text_config.mhc_num_residual_streams == 4
    assert text_config.mla_nope is True
    assert config.vision_config.rms_norm_eps == 1e-6


def test_glm5_next_rejects_unknown_layer_type():
    with pytest.raises(ValueError, match="Unsupported layer_types"):
        Glm5NextTextConfig(
            num_hidden_layers=1,
            layer_types=["unknown_attention"],
        )


def test_glm5_next_nested_config_roundtrip(tmp_path):
    config = Glm5NextConfig(
        text_config=_text_config(),
        vision_config={"rms_norm_eps": 1e-5},
    )

    config.save_pretrained(tmp_path)
    loaded = Glm5NextConfig.from_pretrained(tmp_path)

    assert loaded.text_config.layer_types == config.text_config.layer_types
    assert loaded.text_config.num_experts_per_token == 8
    assert loaded.text_config.mhc_num_residual_streams == 4
    assert loaded.text_config.mla_nope is True
    assert loaded.vision_config.rms_norm_eps == 1e-6
