"""
IR Model Package

This package contains model architectures and modules for
handwriting recognition and stroke-based sequence modeling.

Modules:
    - context_tracker: Full Context Tracker model with SMM, TPM, visual encoder
    - image_to_text: Encoder-Decoder and Decoder-Only recognition models
    - positional_embedding: Token-to-handwriting position mapping
    - edit_manager: Inference-time edit operations
    - module: Low-level building blocks (attention, RoPE, etc.)
"""

from . import module
from .context_tracker import (
    ContextTrackerConfig,
    ContextTrackerModel,
    create_context_tracker,
)
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
from .tree_to_latex import (
    RelationType,
    TreeNode,
    ExpressionTree,
    TreeToLatex,
    SpatialRelationInferrer,
    tokens_and_actions_to_latex,
)

__all__ = [
    'module',
    # Context Tracker
    'ContextTrackerConfig',
    'ContextTrackerModel',
    'create_context_tracker',
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
    # Tree to LaTeX Conversion
    'RelationType',
    'TreeNode',
    'ExpressionTree',
    'TreeToLatex',
    'SpatialRelationInferrer',
    'tokens_and_actions_to_latex',
]

