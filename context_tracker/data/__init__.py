"""
Context Tracker Data Loaders

- MathWritingAtomic: Atomic symbol and chunk loader from MathWriting dataset
- ChunkRenderer: Render strokes to images with optional artifacts
- CompositionalAugmentor: Create edit scenarios from chunks
"""

from pathlib import Path

from .mathwriting_atomic import (
    MathWritingAtomic,
    MathWritingChunk,
    StrokeBBox,
    Stroke,
    ChunkRenderer,
    CompositionalAugmentor,
    SymbolMaskingAugmentor,
    load_mathwriting_splits,
)

# Default MathWriting data path (relative to this file)
MATHWRITING_ROOT = Path(__file__).parent / "mathwriting" / "mathwriting-2024"

# Alternative paths to check
MATHWRITING_PATHS = [
    MATHWRITING_ROOT,
    Path(__file__).parent / "mathwriting",
    Path("data/mathwriting/mathwriting-2024"),
    Path("data/mathwriting"),
]

def get_mathwriting_path() -> Path:
    """Get the MathWriting data path, checking multiple locations."""
    for path in MATHWRITING_PATHS:
        if path.exists() and (path / "synthetic").exists():
            return path
    raise FileNotFoundError(
        f"MathWriting dataset not found. Checked:\n" + 
        "\n".join(f"  - {p}" for p in MATHWRITING_PATHS)
    )

__all__ = [
    'MathWritingAtomic',
    'MathWritingChunk', 
    'StrokeBBox',
    'Stroke',
    'ChunkRenderer',
    'CompositionalAugmentor',
    'SymbolMaskingAugmentor',
    'load_mathwriting_splits',
    'MATHWRITING_ROOT',
    'get_mathwriting_path',
]

