"""
Compositional Data Augmentation for LaTeX Editing

This package implements chunk-based data augmentation for training
LaTeX editing models. The key insight is that by composing atomic
expression chunks, we can generate exponentially more training data.

Modules:
    - latex_preprocessing: Normalize incomplete LaTeX
    - symbol_extraction: Extract symbol bboxes from rendered PDFs
    - chunk_types: ExpressionChunk and ChunkPool data structures
    - training_example: ComposedTrainingExample output format
    - composer: ContextComposer for building training data

Usage:
    from augmentation import ChunkPool, ContextComposer, build_training_dataset
    from synthetic_case_generator import CaseGenerator
    
    # Quick: use convenience function
    pool, examples = build_training_dataset(
        chunks_per_depth=100,
        num_training_examples=5000,
    )
    
    # Or step-by-step:
    generator = CaseGenerator(seed=42)
    pool = ChunkPool.from_case_generator(generator, chunks_per_depth=50)
    
    composer = ContextComposer(pool)
    examples = composer.compose_batch(num_examples=1000)
"""

# Core data structures
from .chunk_types import ExpressionChunk, ChunkPool

# Training example format
from .training_example import ComposedTrainingExample

# Composition logic
from .composer import ContextComposer, build_training_dataset, render_chunk_pool

# LaTeX utilities
from .latex_preprocessing import (
    normalize_incomplete_latex,
    can_render_latex,
    generate_incomplete_training_examples,
)

# Symbol extraction
from .symbol_extraction import (
    extract_symbol_positions_from_pdf,
    find_symbol_bbox_in_latex,
    get_all_symbol_bboxes,
    create_symbol_crop,
)

__all__ = [
    # Core types
    "ExpressionChunk",
    "ChunkPool",
    "ComposedTrainingExample",
    # Composer
    "ContextComposer",
    "build_training_dataset",
    "render_chunk_pool",
    # LaTeX utils
    "normalize_incomplete_latex",
    "can_render_latex",
    "generate_incomplete_training_examples",
    # Symbol extraction
    "extract_symbol_positions_from_pdf",
    "find_symbol_bbox_in_latex",
    "get_all_symbol_bboxes",
    "create_symbol_crop",
]

