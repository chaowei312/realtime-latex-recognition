"""
Image-Text Demo Model Package

Re-exports model components from context_tracker.model for the toy demo.
This keeps the demo self-contained while using the main project's models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Re-export from context_tracker.model
from context_tracker.model import (
    # Encoder-Decoder
    ImageToTextConfig,
    ImageToTextModel,
    StrokeClassifier,
    CLSAggregator,
    PatchEmbedding,
    create_model_from_config,
    create_stroke_classifier,
    # Decoder-Only
    DecoderOnlyConfig,
    DecoderOnlyImageToText,
    VisualTokenizer,
    create_decoder_only_model,
)

# Re-export modules
from context_tracker.model.module import (
    # Original modules (for line-level OCR)
    Conv2DEncoder,
    Conv1DEncoder,
    ConvEncoder,
    ConvBlock,
    ResidualConvBlock,
    FlashAttention,
    FlashAttentionDecoder,
    FlashAttentionDecoderStack,
    FlashCrossAttention,
    SwiGLU,
    SwiGLUFFN,
    RoPE2D,
    # New configurable modules (for testing)
    StrokePatchEncoder,
    TrajectoryPatchExtractor,
    MultimodalAttentionMask,
    ConfigurableMultiheadAttention,
    load_encoder_from_config,
    list_available_variants,
    load_attention_from_config,
    list_available_patterns,
)

__all__ = [
    # Encoder-Decoder
    'ImageToTextConfig',
    'ImageToTextModel',
    'StrokeClassifier',
    'CLSAggregator',
    'PatchEmbedding',
    'create_model_from_config',
    'create_stroke_classifier',
    # Decoder-Only
    'DecoderOnlyConfig',
    'DecoderOnlyImageToText',
    'VisualTokenizer',
    'create_decoder_only_model',
    # Modules
    'Conv2DEncoder',
    'Conv1DEncoder',
    'ConvEncoder',
    'ConvBlock',
    'ResidualConvBlock',
    'FlashAttention',
    'FlashAttentionDecoder',
    'FlashAttentionDecoderStack',
    'FlashCrossAttention',
    'SwiGLU',
    'SwiGLUFFN',
    'RoPE2D',
    # Configurable
    'StrokePatchEncoder',
    'TrajectoryPatchExtractor',
    'MultimodalAttentionMask',
    'ConfigurableMultiheadAttention',
    'load_encoder_from_config',
    'list_available_variants',
    'load_attention_from_config',
    'list_available_patterns',
]

