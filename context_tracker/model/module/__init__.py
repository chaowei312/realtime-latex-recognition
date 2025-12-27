"""
Model Modules

Core building blocks for the model architecture:
- SwiGLU: Swish-Gated Linear Unit activation
- RoPE2D: 2D Rotary Position Embedding
- FlashAttention: Memory-efficient attention with causal masking
- ConvEncoder: Convolutional feature encoder
- StrokePatchEncoder: Configurable stroke patch encoder (from JSON config)
- ConfigurableAttention: Attention with configurable mask patterns (from JSON config)
- VisualPatchEncoder: Stacked conv encoder for full images → patch embeddings
- StrokeModificationModule: Multi-head stroke selection gate (SMM)
- RelationshipTensor: LaTeX parsing to relationship tensor for position prediction
- TreeAwarePositionModule: Tree-aware position prediction during AR (TPM)
- KCacheManager: K cache management with append-only and recomputation
- ARDecoderWithStrokeRemoval: Option B AR decoder with incremental stroke removal
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
from .visual_encoder import (
    VisualPatchEncoder,
    StackedConvStem,
    PositionalEmbedding2D,
    PatchStrokeMapper,
    create_visual_encoder,
)
from .stroke_modification import (
    StrokeModificationModule,
    StrokeModificationLoss,
    create_stroke_modification_module,
)
from .relationship_tensor import (
    RelationType,
    ParsedLatex,
    ParsedNode,
    LatexRelationshipParser,
    RelationshipTensorModule,
    parse_latex,
    visualize_relationship_tensor,
    visualize_tree,
)
from .position_head import (
    TreeAwarePositionModule,
    TPMLoss,
    KCacheManager,
    ARDecoderWithStrokeRemoval,
    create_tpm,
    create_cache_manager,
    # Aliases
    PositionHead,
    PositionHeadLoss,
    create_position_head,
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
    # Visual Patch Encoder
    'VisualPatchEncoder',
    'StackedConvStem',
    'PositionalEmbedding2D',
    'PatchStrokeMapper',
    'create_visual_encoder',
    # Stroke Modification Module (SMM)
    'StrokeModificationModule',
    'StrokeModificationLoss',
    'create_stroke_modification_module',
    # Relationship Tensor
    'RelationType',
    'ParsedLatex',
    'ParsedNode',
    'LatexRelationshipParser',
    'RelationshipTensorModule',
    'parse_latex',
    'visualize_relationship_tensor',
    'visualize_tree',
    # Tree-aware Position Module (TPM)
    'TreeAwarePositionModule',
    'TPMLoss',
    'KCacheManager',
    'ARDecoderWithStrokeRemoval',
    'create_tpm',
    'create_cache_manager',
    # Aliases
    'PositionHead',
    'PositionHeadLoss',
    'create_position_head',
]

