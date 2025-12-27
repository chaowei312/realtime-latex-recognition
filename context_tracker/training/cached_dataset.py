"""
Cached Dataset for fast training.

Loads pre-computed training samples from disk instead of generating on-the-fly.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CachedEditDataset(Dataset):
    """
    Dataset that loads pre-computed training samples from disk.
    
    Much faster than EditDataset since no on-the-fly rendering.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'synthetic',
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Directory containing prepared data
            split: 'synthetic' for training, 'valid' for validation
            max_samples: Optional limit on samples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load metadata
        metadata_path = self.data_dir / f"{split}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                f"Run prepare_data.py first to generate cached data."
            )
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Find chunk files
        self.chunk_files = sorted(self.data_dir.glob(f"{split}_chunk_*.pkl"))
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files found for split '{split}'")
        
        # Load all samples into memory
        self.samples = []
        for chunk_file in self.chunk_files:
            with open(chunk_file, 'rb') as f:
                chunk_samples = pickle.load(f)
                self.samples.extend(chunk_samples)
                
            if max_samples and len(self.samples) >= max_samples:
                self.samples = self.samples[:max_samples]
                break
        
        print(f"Loaded {len(self.samples)} samples from {len(self.chunk_files)} chunks")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert numpy arrays back to tensors
        result = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value)
            else:
                result[key] = value
        
        return result
    
    @property
    def vocab_size(self) -> int:
        return self.metadata.get('vocab_size', 118)


def collate_cached_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for CachedEditDataset.
    
    Same as collate_edit_batch but works with cached data.
    """
    batch_size = len(batch)
    
    # Get max lengths
    max_context_len = max(b['context_ids'].shape[0] for b in batch)
    max_target_len = max(b['target_tokens'].shape[0] for b in batch)
    max_strokes = max(b['stroke_labels'].shape[1] for b in batch)
    
    # Pad context_ids
    context_ids = torch.zeros(batch_size, max_context_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ctx_len = b['context_ids'].shape[0]
        context_ids[i, :ctx_len] = b['context_ids']
    
    # Stack images (already same size)
    images = torch.stack([b['image'] for b in batch])
    
    # Pad target_tokens
    target_tokens = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    for i, b in enumerate(batch):
        t_len = b['target_tokens'].shape[0]
        target_tokens[i, :t_len] = b['target_tokens']
    
    # Pad target_actions
    target_actions = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    for i, b in enumerate(batch):
        t_len = b['target_actions'].shape[0]
        target_actions[i, :t_len] = b['target_actions']
    
    # Pad stroke_labels [B, T, S]
    stroke_labels = torch.zeros(batch_size, max_target_len, max_strokes)
    for i, b in enumerate(batch):
        t_len, s_len = b['stroke_labels'].shape
        stroke_labels[i, :t_len, :s_len] = b['stroke_labels']
    
    # Pad active_strokes [B, S]
    active_strokes = torch.zeros(batch_size, max_strokes, dtype=torch.bool)
    for i, b in enumerate(batch):
        s_len = b['num_strokes']
        active_strokes[i, :s_len] = True
    
    # Pad is_target_group [B, S]
    is_target_group = torch.zeros(batch_size, max_strokes, dtype=torch.bool)
    for i, b in enumerate(batch):
        s_len = b['is_target_group'].shape[0]
        is_target_group[i, :s_len] = b['is_target_group']
    
    # Collect stroke indices
    stroke_indices = [b['stroke_indices'] for b in batch]
    
    # Sequence lengths
    seq_lens = torch.tensor([b['context_ids'].shape[0] for b in batch])
    
    return {
        'context_ids': context_ids,
        'images': images,
        'target_tokens': target_tokens,
        'target_actions': target_actions,
        'stroke_labels': stroke_labels,
        'active_strokes': active_strokes,
        'is_target_group': is_target_group,
        'stroke_indices': stroke_indices,
        'seq_lens': seq_lens,
    }


def create_cached_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from cached data.
    
    Args:
        data_dir: Directory containing prepared data
        batch_size: Batch size
        num_workers: DataLoader workers
        max_train_samples: Limit training samples
        max_val_samples: Limit validation samples
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = CachedEditDataset(
        data_dir,
        split='synthetic',
        max_samples=max_train_samples
    )
    
    val_dataset = CachedEditDataset(
        data_dir,
        split='valid',
        max_samples=max_val_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_cached_batch,
        pin_memory=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_cached_batch,
        pin_memory=num_workers > 0
    )
    
    return train_loader, val_loader


# Test
if __name__ == '__main__':
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/prepared'
    
    print(f"Testing CachedEditDataset from {data_dir}")
    
    try:
        dataset = CachedEditDataset(data_dir, split='synthetic', max_samples=100)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print("\nSample keys:", list(sample.keys()))
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")
        
        # Test dataloader
        loader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=collate_cached_batch
        )
        batch = next(iter(loader))
        print("\nBatch keys:", list(batch.keys()))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun prepare_data.py first to generate cached data.")

