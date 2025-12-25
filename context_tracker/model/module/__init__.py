"""
Model Modules

Core building blocks for the model architecture:
- SwiGLU: Swish-Gated Linear Unit activation
- RoPE2D: 2D Rotary Position Embedding
- FlashAttention: Memory-efficient attention with causal masking
- ConvEncoder: Convolutional feature encoder
- StrokePatchEncoder: Configurable stroke patch encoder (from JSON config)
- ConfigurableAttention: Attention with configurable mask patterns (from JSON config)
"""

from .swiglu import SwiGLU, SwiGLUFFN
from .rope_2d import RoPE2D, RoPE2DForStrokes, apply_rope_2d, precompute_freqs_2d
from .flash_attention import FlashAttention, FlashAttentionDecoder, FlashAttentionDecoderStack, FlashCrossAttention
from .conv_encoder import ConvEncoder, ConvBlock, ResidualConvBlock, Conv1DEncoder, Conv2DEncoder
from .stroke_patch_encoder import (
    StrokePatchEncoder,
    TrajectoryPatchExtractor,
    ConfigurableConvBlock,
    DepthwiseSeparableConv,
    load_encoder_from_config,
    list_available_variants,
)
from .configurable_attention import (
    MultimodalAttentionMask,
    ConfigurableMultiheadAttention,
    StrokeGroupedAttention,
    load_attention_from_config,
    list_available_patterns,
    MaskType,
    TokenType,
)

__all__ = [
    # SwiGLU
    'SwiGLU',
    'SwiGLUFFN',
    # 2D RoPE
    'RoPE2D',
    'RoPE2DForStrokes',
    'apply_rope_2d',
    'precompute_freqs_2d',
    # Flash Attention
    'FlashAttention',
    'FlashAttentionDecoder',
    'FlashAttentionDecoderStack',
    'FlashCrossAttention',
    # Conv Encoder
    'ConvEncoder',
    'ConvBlock',
    'ResidualConvBlock',
    'Conv1DEncoder',
    'Conv2DEncoder',
    # Stroke Patch Encoder (Configurable)
    'StrokePatchEncoder',
    'TrajectoryPatchExtractor',
    'ConfigurableConvBlock',
    'DepthwiseSeparableConv',
    'load_encoder_from_config',
    'list_available_variants',
    # Configurable Attention
    'MultimodalAttentionMask',
    'ConfigurableMultiheadAttention',
    'StrokeGroupedAttention',
    'load_attention_from_config',
    'list_available_patterns',
    'MaskType',
    'TokenType',
]

