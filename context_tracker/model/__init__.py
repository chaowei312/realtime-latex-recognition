"""
IR Model Package

This package contains model architectures and modules for
handwriting recognition and stroke-based sequence modeling.

Modules:
    - image_to_text: Encoder-Decoder and Decoder-Only recognition models
    - positional_embedding: Token-to-handwriting position mapping
    - edit_manager: Inference-time edit operations
    - module: Low-level building blocks (attention, RoPE, etc.)
"""

from . import module
from .image_to_text import (
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
from .positional_embedding import (
    BoundingBox,
    TokenPosition,
    TokenHandwritingMapping,
    LaTeXContext,
    PositionalEmbeddingComputer,
    compute_token_positions_from_latex_render,
)
from .edit_manager import (
    EditType,
    StrokeRecord,
    TokenRecord,
    EditRegion,
    EditManager,
)

__all__ = [
    'module',
    # Encoder-Decoder Image-to-Text
    'ImageToTextConfig',
    'ImageToTextModel',
    'StrokeClassifier',
    'CLSAggregator',
    'PatchEmbedding',
    'create_model_from_config',
    'create_stroke_classifier',
    # Decoder-Only Image-to-Text
    'DecoderOnlyConfig',
    'DecoderOnlyImageToText',
    'VisualTokenizer',
    'create_decoder_only_model',
    # Positional Embedding
    'BoundingBox',
    'TokenPosition',
    'TokenHandwritingMapping',
    'LaTeXContext',
    'PositionalEmbeddingComputer',
    'compute_token_positions_from_latex_render',
    # Edit Manager
    'EditType',
    'StrokeRecord',
    'TokenRecord',
    'EditRegion',
    'EditManager',
]

