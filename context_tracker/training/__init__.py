"""
Training utilities for Context Tracker.

- train_step: Single training step with Option B (incremental stroke removal)
- Trainer: Full training loop with logging, checkpointing, evaluation
- EditDataset: PyTorch Dataset for compositional augmentation training
- SimpleTokenizer: Basic LaTeX tokenizer for prototyping
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    train_step,
    evaluate,
)
from .dataset import (
    EditDataset,
    EditSample,
    SimpleTokenizer,
    collate_edit_batch,
    create_dataloaders,
)

__all__ = [
    # Trainer
    'Trainer',
    'TrainingConfig', 
    'train_step',
    'evaluate',
    # Dataset
    'EditDataset',
    'EditSample',
    'SimpleTokenizer',
    'collate_edit_batch',
    'create_dataloaders',
]

