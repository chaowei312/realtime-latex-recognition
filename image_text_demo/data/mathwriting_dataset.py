"""
MathWriting Dataset Loader

Loads stroke trajectory data from MathWriting InkML files for
trajectory-based autoregressive text recognition.

MathWriting format:
- InkML files with <trace> elements containing "x y t, x y t, ..." points
- LaTeX ground truth in <annotation type="normalizedLabel">
"""

import os
import re
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MathWritingSample:
    """Single MathWriting sample."""
    sample_id: str
    label: str  # LaTeX expression
    normalized_label: str
    strokes: List[List[Tuple[float, float, float]]]  # List of strokes, each is [(x, y, t), ...]
    ink_method: str  # 'human' or 'synthetic'
    

def parse_inkml(filepath: Path) -> Optional[MathWritingSample]:
    """Parse an InkML file into a MathWritingSample."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Handle namespace
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        
        # Get annotations
        sample_id = filepath.stem
        label = ""
        normalized_label = ""
        ink_method = "human"
        
        for ann in root.findall('.//ink:annotation', ns):
            ann_type = ann.get('type')
            if ann_type == 'label':
                label = ann.text or ""
            elif ann_type == 'normalizedLabel':
                normalized_label = ann.text or ""
            elif ann_type == 'inkCreationMethod':
                ink_method = ann.text or "human"
            elif ann_type == 'sampleId':
                sample_id = ann.text or sample_id
        
        # Parse strokes
        strokes = []
        for trace in root.findall('.//ink:trace', ns):
            if trace.text:
                points = []
                # Format: "x y t, x y t, ..."
                for point_str in trace.text.strip().split(','):
                    parts = point_str.strip().split()
                    if len(parts) >= 3:
                        x, y, t = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append((x, y, t))
                if points:
                    strokes.append(points)
        
        if not strokes or not normalized_label:
            return None
            
        return MathWritingSample(
            sample_id=sample_id,
            label=label,
            normalized_label=normalized_label,
            strokes=strokes,
            ink_method=ink_method,
        )
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


class MathWritingDataset(Dataset):
    """
    Dataset for MathWriting stroke trajectory recognition.
    
    Uses LAZY LOADING - only loads/parses files when needed for each batch.
    No RAM explosion even with 626K samples!
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        include_synthetic: bool = True,
        max_strokes: int = 50,
        max_text_len: int = 128,
        canvas_size: Tuple[int, int] = (128, 512),
        patch_size: int = 16,
        sample_interval: int = 4,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_strokes = max_strokes
        self.max_text_len = max_text_len
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.sample_interval = sample_interval
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        self.pad_id = self.vocab.get('<PAD>', 0)
        self.bos_id = self.vocab.get('<BOS>', 1)
        self.eos_id = self.vocab.get('<EOS>', 2)
        self.unk_id = self.vocab.get('<UNK>', 3)
        
        # LAZY LOADING: Only store file paths, not parsed data!
        self.file_paths = self._collect_file_paths(include_synthetic, max_samples)
        print(f"MathWriting {split}: {len(self.file_paths)} samples (lazy loading)")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from common LaTeX tokens."""
        # Start with special tokens
        vocab = {
            '<PAD>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
        }
        
        # Add common characters
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "+-*/=()[]{}|<>.,;:!?@#$%^&*_~`'\"\\ "
        )
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)
        
        # Add common LaTeX commands (as single tokens for simplicity)
        # We'll treat backslash sequences as individual characters
        
        return vocab
    
    def _collect_file_paths(self, include_synthetic: bool, max_samples: int = None) -> List[Path]:
        """Collect file paths only - NO parsing, NO loading into RAM."""
        file_paths = []
        
        # Determine directories
        dirs_to_load = [self.data_dir / self.split]
        if include_synthetic and self.split == "train":
            dirs_to_load.append(self.data_dir / "synthetic")
        
        for dir_path in dirs_to_load:
            if not dir_path.exists():
                print(f"Warning: {dir_path} does not exist")
                continue
            
            inkml_files = sorted(dir_path.glob("*.inkml"))
            print(f"Found {len(inkml_files)} files in {dir_path.name}")
            
            # Limit if requested
            if max_samples:
                remaining = max_samples - len(file_paths)
                if remaining <= 0:
                    break
                inkml_files = inkml_files[:remaining]
            
            file_paths.extend(inkml_files)
        
        return file_paths
    
    def _render_strokes(self, strokes: List[List[Tuple[float, float, float]]]) -> torch.Tensor:
        """Render strokes to canvas, normalizing coordinates."""
        H, W = self.canvas_size
        canvas = torch.ones(1, H, W)
        
        # Find bounding box of all strokes
        all_points = [(x, y) for stroke in strokes for x, y, _ in stroke]
        if not all_points:
            return canvas
        
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add margin
        margin = 10
        width = max(max_x - min_x, 1)
        height = max(max_y - min_y, 1)
        
        # Scale to fit canvas
        scale_x = (W - 2 * margin) / width
        scale_y = (H - 2 * margin) / height
        scale = min(scale_x, scale_y)
        
        # Render each stroke
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x0 = int((stroke[i][0] - min_x) * scale + margin)
                y0 = int((stroke[i][1] - min_y) * scale + margin)
                x1 = int((stroke[i+1][0] - min_x) * scale + margin)
                y1 = int((stroke[i+1][1] - min_y) * scale + margin)
                
                self._draw_line(canvas, x0, y0, x1, y1)
        
        return canvas
    
    def _draw_line(self, canvas: torch.Tensor, x0: int, y0: int, x1: int, y1: int):
        """Draw anti-aliased line on canvas."""
        H, W = canvas.shape[1], canvas.shape[2]
        num_steps = max(int(abs(x1 - x0) + abs(y1 - y0)), 1)
        
        for t in range(num_steps + 1):
            px = int(x0 + (x1 - x0) * t / num_steps)
            py = int(y0 + (y1 - y0) * t / num_steps)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        canvas[0, ny, nx] = 0.0
    
    def _extract_patches(
        self,
        strokes: List[List[Tuple[float, float, float]]],
        canvas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract patches along stroke trajectories."""
        H, W = self.canvas_size
        half = self.patch_size // 2
        
        # Get normalization params (same as rendering)
        all_points = [(x, y) for stroke in strokes for x, y, _ in stroke]
        if not all_points:
            return torch.ones(1, 1, self.patch_size, self.patch_size), torch.zeros(1, dtype=torch.long)
        
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        margin = 10
        width = max(max_x - min_x, 1)
        height = max(max_y - min_y, 1)
        scale_x = (W - 2 * margin) / width
        scale_y = (H - 2 * margin) / height
        scale = min(scale_x, scale_y)
        
        all_patches = []
        all_stroke_ids = []
        
        for stroke_idx, stroke in enumerate(strokes[:self.max_strokes]):
            for i in range(0, len(stroke), self.sample_interval):
                px = int((stroke[i][0] - min_x) * scale + margin)
                py = int((stroke[i][1] - min_y) * scale + margin)
                
                x1 = max(0, px - half)
                y1 = max(0, py - half)
                x2 = min(W, px + half)
                y2 = min(H, py + half)
                
                patch = canvas[:, y1:y2, x1:x2].clone()
                
                if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
                    padded = torch.ones(1, self.patch_size, self.patch_size)
                    padded[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                
                all_patches.append(patch)
                all_stroke_ids.append(stroke_idx)
        
        if not all_patches:
            return torch.ones(1, 1, self.patch_size, self.patch_size), torch.zeros(1, dtype=torch.long)
        
        return torch.stack(all_patches), torch.tensor(all_stroke_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """Lazy load: parse file and extract patches on-demand."""
        filepath = self.file_paths[idx]
        
        # Parse InkML file (only when needed)
        sample = parse_inkml(filepath)
        
        if sample is None:
            # Return dummy data for failed parses
            return self._get_dummy_sample()
        
        # Render strokes and extract patches
        canvas = self._render_strokes(sample.strokes)
        patches, stroke_ids = self._extract_patches(sample.strokes, canvas)
        
        # Tokenize LaTeX label
        text = sample.normalized_label
        input_ids, target_ids = self._tokenize_text(text)
        
        return {
            'patches': patches,
            'stroke_ids': stroke_ids,
            'num_patches': patches.shape[0],
            'num_strokes': len(sample.strokes),
            'input_ids': input_ids,
            'target_ids': target_ids,
            'text': text,
            'text_len': len(text),
        }
    
    def _get_dummy_sample(self) -> Dict:
        """Return dummy sample for failed parses."""
        return {
            'patches': torch.ones(1, 1, self.patch_size, self.patch_size),
            'stroke_ids': torch.zeros(1, dtype=torch.long),
            'num_patches': 1,
            'num_strokes': 0,
            'input_ids': torch.zeros(self.max_text_len, dtype=torch.long),
            'target_ids': torch.zeros(self.max_text_len, dtype=torch.long),
            'text': '',
            'text_len': 0,
        }
    
    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize LaTeX text character by character."""
        indices = []
        for c in text:
            indices.append(self.vocab.get(c, self.unk_id))
        
        indices = indices[:self.max_text_len - 2]
        
        input_ids = [self.bos_id] + indices
        target_ids = indices + [self.eos_id]
        
        input_ids = input_ids + [self.pad_id] * (self.max_text_len - len(input_ids))
        target_ids = target_ids + [self.pad_id] * (self.max_text_len - len(target_ids))
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def get_mathwriting_datasets(
    data_dir: str,
    include_synthetic: bool = True,
    **kwargs,
) -> Tuple[MathWritingDataset, MathWritingDataset, MathWritingDataset]:
    """Get train, valid, test datasets."""
    
    # Find the actual data directory (prefer full over excerpt)
    data_path = Path(data_dir)
    if (data_path / "mathwriting-2024" / "train").exists():
        data_path = data_path / "mathwriting-2024"
        print(f"Using FULL MathWriting dataset")
    elif (data_path / "mathwriting-2024-excerpt" / "train").exists():
        data_path = data_path / "mathwriting-2024-excerpt"
        print(f"Using MathWriting EXCERPT (for testing)")
    else:
        raise FileNotFoundError(f"No MathWriting data found in {data_dir}")
    
    # For validation/test, we can use fewer samples
    max_samples = kwargs.pop('max_samples', None)
    val_max = min(max_samples // 10, 5000) if max_samples else None
    
    train_ds = MathWritingDataset(
        str(data_path), 
        split="train", 
        include_synthetic=include_synthetic,
        max_samples=max_samples,
        **kwargs
    )
    
    valid_ds = MathWritingDataset(
        str(data_path), 
        split="valid",
        include_synthetic=False,
        max_samples=val_max,
        **kwargs
    )
    
    test_ds = MathWritingDataset(
        str(data_path), 
        split="test",
        include_synthetic=False,
        max_samples=val_max,
        **kwargs
    )
    
    # Share vocabulary
    valid_ds.vocab = train_ds.vocab
    valid_ds.idx_to_char = train_ds.idx_to_char
    test_ds.vocab = train_ds.vocab
    test_ds.idx_to_char = train_ds.idx_to_char
    
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Test loading
    data_dir = Path(__file__).parent / "mathwriting"
    
    train_ds, valid_ds, test_ds = get_mathwriting_datasets(str(data_dir))
    
    print(f"\nTrain: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
    print(f"Vocab size: {train_ds.vocab_size}")
    
    # Show sample
    sample = train_ds[0]
    print(f"\nSample:")
    print(f"  Text: {sample['text'][:50]}...")
    print(f"  Patches: {sample['patches'].shape}")
    print(f"  Num strokes: {sample['num_strokes']}")

