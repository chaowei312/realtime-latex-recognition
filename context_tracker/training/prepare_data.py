#!/usr/bin/env python3
"""
Pre-compute training data for faster training.

This script generates all augmented training samples and saves them to disk,
avoiding the expensive on-the-fly stroke rendering during training.

Usage:
    python -m context_tracker.training.prepare_data --output_dir data/prepared --max_samples 10000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

import torch
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from context_tracker.data import MathWritingAtomic, get_mathwriting_path
from context_tracker.training.dataset import EditDataset, SimpleTokenizer


def prepare_single_sample(
    idx: int,
    dataset: EditDataset
) -> Optional[Dict]:
    """Generate a single training sample."""
    try:
        sample = dataset[idx]
        
        # Convert tensors to numpy for efficient storage
        result = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.numpy()
            elif isinstance(value, (int, float, str, list, tuple)):
                result[key] = value
            else:
                result[key] = value
        
        return result
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None


def prepare_dataset(
    output_dir: Path,
    split: str,
    max_samples: int,
    image_size: int,
    samples_per_chunk: int,
    num_workers: int
):
    """Prepare and save dataset to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MathWriting data
    data_path = get_mathwriting_path()
    print(f"Loading MathWriting from: {data_path}")
    
    mw = MathWritingAtomic(
        data_path,
        split=split,
        lazy_load=True,
        load_bboxes=True,
        max_samples=max_samples
    )
    print(f"Loaded {len(mw)} samples from {split} split")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create dataset (this doesn't generate samples yet)
    dataset = EditDataset(
        mw, tokenizer,
        image_size=image_size,
        max_context_len=64,
        min_chunks=2,
        max_chunks=3
    )
    
    total_samples = len(dataset)
    print(f"Will generate {total_samples} augmented samples")
    
    # Generate samples
    samples = []
    chunk_idx = 0
    
    for i in tqdm(range(total_samples), desc=f"Generating {split} samples"):
        sample = prepare_single_sample(i, dataset)
        if sample is not None:
            samples.append(sample)
        
        # Save chunk when full
        if len(samples) >= samples_per_chunk:
            chunk_path = output_dir / f"{split}_chunk_{chunk_idx:04d}.pkl"
            with open(chunk_path, 'wb') as f:
                pickle.dump(samples, f)
            print(f"Saved {len(samples)} samples to {chunk_path}")
            samples = []
            chunk_idx += 1
    
    # Save remaining samples
    if samples:
        chunk_path = output_dir / f"{split}_chunk_{chunk_idx:04d}.pkl"
        with open(chunk_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Saved {len(samples)} samples to {chunk_path}")
        chunk_idx += 1
    
    # Save metadata
    metadata = {
        'split': split,
        'total_samples': total_samples,
        'num_chunks': chunk_idx,
        'samples_per_chunk': samples_per_chunk,
        'image_size': image_size,
        'vocab_size': len(tokenizer),
        'tokenizer_vocab': tokenizer.symbol_to_id,
    }
    
    metadata_path = output_dir / f"{split}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return total_samples


def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--output_dir', type=str, default='data/prepared',
                        help='Output directory for prepared data')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples from MathWriting (None = all)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--samples_per_chunk', type=int, default=10000,
                        help='Samples per chunk file')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for parallel processing')
    parser.add_argument('--train_only', action='store_true',
                        help='Only prepare training data')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Prepare training data
    print("\n" + "="*60)
    print("Preparing TRAINING data")
    print("="*60)
    train_count = prepare_dataset(
        output_dir,
        split='synthetic',
        max_samples=args.max_samples,
        image_size=args.image_size,
        samples_per_chunk=args.samples_per_chunk,
        num_workers=args.num_workers
    )
    
    if not args.train_only:
        # Prepare validation data
        print("\n" + "="*60)
        print("Preparing VALIDATION data")
        print("="*60)
        val_max = min(5000, args.max_samples) if args.max_samples else 5000
        val_count = prepare_dataset(
            output_dir,
            split='valid',
            max_samples=val_max,
            image_size=args.image_size,
            samples_per_chunk=args.samples_per_chunk,
            num_workers=args.num_workers
        )
    
    print("\n" + "="*60)
    print("DONE!")
    print(f"Training samples: {train_count}")
    if not args.train_only:
        print(f"Validation samples: {val_count}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

