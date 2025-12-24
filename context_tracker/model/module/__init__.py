"""
Model Modules

Core building blocks for the model architecture:
- SwiGLU: Swish-Gated Linear Unit activation
- RoPE2D: 2D Rotary Position Embedding
- FlashAttention: Memory-efficient attention with causal masking
- ConvEncoder: Convolutional feature encoder
"""

from .swiglu import SwiGLU, SwiGLUFFN
from .rope_2d import RoPE2D, RoPE2DForStrokes, apply_rope_2d, precompute_freqs_2d
from .flash_attention import FlashAttention, FlashAttentionDecoder, FlashAttentionDecoderStack, FlashCrossAttention
from .conv_encoder import ConvEncoder, ConvBlock, ResidualConvBlock, Conv1DEncoder, Conv2DEncoder

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
]

