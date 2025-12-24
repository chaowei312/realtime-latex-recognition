"""
Scripts for stroke-based handwriting synthesis and training data generation.

Modules:
    stroke_renderer: GPU-accelerated stroke rendering with captured data
    stroke_dataset: Dataset generation for training
    latex_renderer: LaTeX expression to image rendering
"""

# Imports are done lazily to avoid circular dependencies
# Use: 
#   from scripts.stroke_renderer import StrokeRenderer
#   from scripts.latex_renderer import render_latex

__all__ = [
    'stroke_renderer',
    'stroke_dataset',
    'latex_renderer',
]

