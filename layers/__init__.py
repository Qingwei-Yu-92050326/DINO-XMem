# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .attention import CausalSelfAttention, LinearKMaskedBias, SelfAttention
from .block import CausalSelfAttentionBlock, SelfAttentionBlock
from .ffn_layers import Mlp, SwiGLUFFN
try:  # pragma: no cover
    from .fp8_linear import convert_linears_to_fp8
except Exception as e:  # pragma: no cover
    def convert_linears_to_fp8(*args, **kwargs):
        raise RuntimeError(
            "convert_linears_to_fp8 is unavailable in this environment. "
            "Please use a newer PyTorch version / proper GPU stack if you need fp8 support."
        ) from e
from .layer_scale import LayerScale
from .patch_embed import PatchEmbed
from .rms_norm import RMSNorm
from .rope_position_encoding import RopePositionEmbedding
