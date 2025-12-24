"""
IAM On-Line Handwriting Database: Stroke -> Letter Dataset Builder

This script constructs a dataset where:
- Input: Single stroke trajectory [(x, y, t), ...]
- Output: Letter/character the stroke belongs to

The mapping is done using spatial segmentation heuristics since
the original data doesn't have per-stroke character labels.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json
import pickle
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Point:
    x: float
    y: float
    time: float


@dataclass
class Stroke:
    points: List[Point]
    start_time: float
    end_time: float
    
    @property
    def min_x(self) -> float:
        return min(p.x for p in self.points) if self.points else 0
    
    @property
    def max_x(self) -> float:
        return max(p.x for p in self.points) if self.points else 0
    
    @property
    def min_y(self) -> float:
        return min(p.y for p in self.points) if self.points else 0
    
    @property
    def max_y(self) -> float:
        return max(p.y for p in self.points) if self.points else 0
    
    @property
    def center_x(self) -> float:
        return np.mean([p.x for p in self.points]) if self.points else 0
    
    @property
    def center_y(self) -> float:
        return np.mean([p.y for p in self.points]) if self.points else 0
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    def to_array(self) -> np.ndarray:
        """Convert stroke to numpy array of shape (N, 3) for [x, y, t]"""
        return np.array([[p.x, p.y, p.time] for p in self.points])
    
    def to_normalized_array(self) -> np.ndarray:
        """Normalize stroke coordinates to [0, 1] range"""
        arr = self.to_array()
        if len(arr) == 0:
            return arr
        
        # Normalize x, y to [0, 1]
        xy = arr[:, :2]
        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_range = xy_max - xy_min
        xy_range[xy_range == 0] = 1  # Avoid division by zero
        
        arr[:, :2] = (xy - xy_min) / xy_range
        
        # Normalize time to start from 0
        arr[:, 2] = arr[:, 2] - arr[0, 2]
        
        return arr


@dataclass
class StrokeLetterSample:
    """A single sample in our dataset"""
    stroke_id: str
    stroke_points: np.ndarray  # Shape: (N, 3) for [x, y, t]
    stroke_points_normalized: np.ndarray  # Normalized version
    letter: str
    line_id: str
    stroke_index: int  # Which stroke in the line
    num_points: int
    # Additional metadata
    confidence: float = 1.0  # How confident we are in the letter assignment


# ============================================================
# Parser
# ============================================================

class IAMOnDBParser:
    """Parse IAM On-Line Handwriting Database files"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.line_strokes_dir = self.data_root / "lineStrokes-all" / "lineStrokes"
        self.ascii_dir = self.data_root / "ascii-all" / "ascii"
    
    def parse_stroke_xml(self, xml_path: str) -> List[Stroke]:
        """Parse stroke XML file and return list of Stroke objects"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return []
        
        strokes = []
        for stroke_elem in root.findall('.//Stroke'):
            points = []
            for point_elem in stroke_elem.findall('Point'):
                try:
                    points.append(Point(
                        x=float(point_elem.get('x', 0)),
                        y=float(point_elem.get('y', 0)),
                        time=float(point_elem.get('time', 0))
                    ))
                except (ValueError, TypeError):
                    continue
            
            if points:  # Only add non-empty strokes
                strokes.append(Stroke(
                    points=points,
                    start_time=float(stroke_elem.get('start_time', 0)),
                    end_time=float(stroke_elem.get('end_time', 0))
                ))
        
        return strokes
    
    def parse_ascii_file(self, ascii_path: str) -> dict:
        """Parse ASCII transcription file"""
        try:
            with open(ascii_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {ascii_path}: {e}")
            return {'ocr': '', 'csr': ''}
        
        result = {'ocr': '', 'csr': ''}
        
        if 'CSR:' in content:
            parts = content.split('CSR:')
            result['ocr'] = parts[0].replace('OCR:', '').strip()
            result['csr'] = parts[1].strip() if len(parts) > 1 else ''
        elif 'OCR:' in content:
            result['ocr'] = content.replace('OCR:', '').strip()
            result['csr'] = result['ocr']
        else:
            # No markers, use content as-is
            result['csr'] = content.strip()
        
        return result
    
    def get_line_text(self, line_id: str) -> Optional[str]:
        """
        Get text for a specific line from ASCII files
        
        line_id format: 'a01-000u-01' 
        - form: 'a01-000u'
        - line number: 01
        """
        # Parse line_id: a01-000u-01 -> group=a01, form_base=000, variant=u, line=01
        parts = line_id.split('-')
        if len(parts) < 3:
            return None
        
        group = parts[0]  # e.g., 'a01'
        form_with_variant = parts[1]  # e.g., '000u' or '000'
        line_num_str = parts[2]  # e.g., '01'
        
        try:
            line_num = int(line_num_str) - 1  # 0-indexed
        except ValueError:
            return None
        
        # Reconstruct form_id
        form_id = f"{group}-{form_with_variant}"
        
        # Try to find the ASCII file
        # Structure: ascii/group/form_base/form_id.txt
        form_base = ''.join(c for c in form_with_variant if c.isdigit())
        form_dir = f"{group}-{form_base}"
        
        ascii_path = self.ascii_dir / group / form_dir / f"{form_id}.txt"
        
        if not ascii_path.exists():
            # Try alternative path structure
            ascii_path = self.ascii_dir / group / f"{group}-{form_with_variant}" / f"{form_id}.txt"
        
        if not ascii_path.exists():
            return None
        
        transcription = self.parse_ascii_file(str(ascii_path))
        
        # Get the specific line
        text = transcription.get('csr', '') or transcription.get('ocr', '')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if 0 <= line_num < len(lines):
            return lines[line_num]
        
        return None
    
    def iter_line_files(self):
        """Iterate through all line stroke XML files"""
        if not self.line_strokes_dir.exists():
            print(f"Directory not found: {self.line_strokes_dir}")
            return
        
        for group_dir in sorted(self.line_strokes_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            
            for form_dir in sorted(group_dir.iterdir()):
                if not form_dir.is_dir():
                    continue
                
                for xml_file in sorted(form_dir.glob("*.xml")):
                    line_id = xml_file.stem  # e.g., 'a01-000u-01'
                    yield line_id, str(xml_file)


# ============================================================
# Character Segmenter
# ============================================================

class StrokeToLetterMapper:
    """
    Map strokes to letters using spatial segmentation.
    
    Strategy:
    1. Sort strokes by horizontal position (left to right)
    2. Identify gaps between stroke groups (word/character boundaries)
    3. Map stroke groups to characters based on position alignment
    """
    
    def __init__(self, 
                 char_gap_threshold: float = 0.08,
                 word_gap_threshold: float = 0.15):
        """
        Args:
            char_gap_threshold: Fraction of line width for character gaps
            word_gap_threshold: Fraction of line width for word gaps (space)
        """
        self.char_gap_threshold = char_gap_threshold
        self.word_gap_threshold = word_gap_threshold
    
    def map_strokes_to_letters(self, 
                                strokes: List[Stroke], 
                                text: str) -> List[Tuple[Stroke, str, float]]:
        """
        Map each stroke to its corresponding letter.
        
        Returns:
            List of (stroke, letter, confidence) tuples
        """
        if not strokes or not text:
            return []
        
        # Remove extra spaces, keep single spaces
        text = ' '.join(text.split())
        
        # Get all characters (including spaces for word boundaries)
        chars = list(text)
        non_space_chars = [c for c in chars if c != ' ']
        
        if not non_space_chars:
            return []
        
        # Sort strokes by center x position (left to right reading order)
        sorted_strokes = sorted(enumerate(strokes), key=lambda x: x[1].center_x)
        
        # Calculate line width
        all_xs = [p.x for s in strokes for p in s.points]
        if not all_xs:
            return []
        
        line_width = max(all_xs) - min(all_xs)
        if line_width == 0:
            line_width = 1
        
        # Group strokes into character clusters
        char_gap = line_width * self.char_gap_threshold
        word_gap = line_width * self.word_gap_threshold
        
        # Create groups based on gaps
        groups = []
        current_group = [sorted_strokes[0]]
        
        for i in range(1, len(sorted_strokes)):
            prev_idx, prev_stroke = sorted_strokes[i - 1]
            curr_idx, curr_stroke = sorted_strokes[i]
            
            # Calculate gap
            gap = curr_stroke.min_x - prev_stroke.max_x
            
            if gap > word_gap:
                # Word boundary - save current group and start new
                groups.append(('char', current_group))
                groups.append(('space', None))
                current_group = [(curr_idx, curr_stroke)]
            elif gap > char_gap:
                # Character boundary
                groups.append(('char', current_group))
                current_group = [(curr_idx, curr_stroke)]
            else:
                # Same character
                current_group.append((curr_idx, curr_stroke))
        
        # Don't forget the last group
        if current_group:
            groups.append(('char', current_group))
        
        # Now align groups with characters
        result = [None] * len(strokes)  # Will store (letter, confidence) for each stroke
        
        char_groups = [g for g in groups if g[0] == 'char']
        
        # Simple alignment: distribute characters across groups
        if len(char_groups) == 0:
            return []
        
        # Calculate how many chars per group
        chars_per_group = len(non_space_chars) / len(char_groups)
        
        char_idx = 0
        for group_idx, (group_type, group_strokes) in enumerate(groups):
            if group_type == 'space':
                continue
            
            # Determine which character(s) this group corresponds to
            # Use proportional mapping
            group_num = sum(1 for g in groups[:group_idx + 1] if g[0] == 'char') - 1
            
            start_char_idx = int(group_num * chars_per_group)
            end_char_idx = int((group_num + 1) * chars_per_group)
            end_char_idx = min(end_char_idx, len(non_space_chars))
            
            if start_char_idx >= len(non_space_chars):
                start_char_idx = len(non_space_chars) - 1
            
            # Get the character(s) for this group
            group_chars = non_space_chars[start_char_idx:end_char_idx]
            if not group_chars:
                group_chars = [non_space_chars[min(start_char_idx, len(non_space_chars) - 1)]]
            
            # Assign to all strokes in this group
            # If multiple characters, assign based on stroke position within group
            for stroke_pos, (orig_idx, stroke) in enumerate(group_strokes):
                if len(group_chars) == 1:
                    letter = group_chars[0]
                    confidence = 0.9
                else:
                    # Multiple chars for this group - distribute by position
                    char_pos = int(stroke_pos / len(group_strokes) * len(group_chars))
                    char_pos = min(char_pos, len(group_chars) - 1)
                    letter = group_chars[char_pos]
                    confidence = 0.7  # Lower confidence for ambiguous cases
                
                result[orig_idx] = (letter, confidence)
        
        # Build final output maintaining original stroke order
        output = []
        for i, stroke in enumerate(strokes):
            if result[i] is not None:
                letter, confidence = result[i]
                output.append((stroke, letter, confidence))
            else:
                # Fallback - shouldn't happen but just in case
                output.append((stroke, '?', 0.1))
        
        return output


# ============================================================
# Dataset Builder
# ============================================================

class StrokeLetterDatasetBuilder:
    """Build the stroke → letter dataset"""
    
    def __init__(self, data_root: str, output_dir: str):
        self.parser = IAMOnDBParser(data_root)
        self.mapper = StrokeToLetterMapper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_dataset(self, 
                      max_lines: int = None,
                      min_confidence: float = 0.5,
                      min_points: int = 3) -> List[Dict]:
        """
        Build the dataset.
        
        Args:
            max_lines: Maximum number of lines to process (None for all)
            min_confidence: Minimum confidence threshold for samples
            min_points: Minimum number of points in a stroke
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        stats = defaultdict(int)
        
        line_files = list(self.parser.iter_line_files())
        if max_lines:
            line_files = line_files[:max_lines]
        
        print(f"Processing {len(line_files)} line files...")
        
        for line_id, xml_path in tqdm(line_files, desc="Building dataset"):
            # Get transcription
            text = self.parser.get_line_text(line_id)
            if not text:
                stats['no_text'] += 1
                continue
            
            # Parse strokes
            strokes = self.parser.parse_stroke_xml(xml_path)
            if not strokes:
                stats['no_strokes'] += 1
                continue
            
            # Map strokes to letters
            try:
                mappings = self.mapper.map_strokes_to_letters(strokes, text)
            except Exception as e:
                stats['mapping_error'] += 1
                continue
            
            # Create samples
            for stroke_idx, (stroke, letter, confidence) in enumerate(mappings):
                # Filter by confidence and point count
                if confidence < min_confidence:
                    stats['low_confidence'] += 1
                    continue
                
                if len(stroke.points) < min_points:
                    stats['too_few_points'] += 1
                    continue
                
                # Skip unknown characters
                if letter == '?':
                    stats['unknown_letter'] += 1
                    continue
                
                stroke_id = f"{line_id}_s{stroke_idx:03d}"
                
                sample = {
                    'stroke_id': stroke_id,
                    'stroke_points': stroke.to_array().tolist(),
                    'stroke_points_normalized': stroke.to_normalized_array().tolist(),
                    'letter': letter,
                    'line_id': line_id,
                    'stroke_index': stroke_idx,
                    'num_points': len(stroke.points),
                    'confidence': confidence,
                    # Additional features
                    'width': stroke.width,
                    'height': stroke.height,
                    'duration': stroke.end_time - stroke.start_time,
                }
                
                samples.append(sample)
                stats['valid_samples'] += 1
                stats[f'letter_{letter}'] += 1
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total valid samples: {stats['valid_samples']}")
        print(f"Skipped - no text: {stats.get('no_text', 0)}")
        print(f"Skipped - no strokes: {stats.get('no_strokes', 0)}")
        print(f"Skipped - low confidence: {stats.get('low_confidence', 0)}")
        print(f"Skipped - too few points: {stats.get('too_few_points', 0)}")
        
        return samples
    
    def save_dataset(self, samples: List[Dict], 
                     format: str = 'both',
                     split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Save the dataset.
        
        Args:
            samples: List of sample dictionaries
            format: 'json', 'pickle', or 'both'
            split_ratio: (train, val, test) ratio
        """
        # Shuffle samples
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        # Split
        n = len(samples)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        
        splits = {
            'train': [samples[i] for i in indices[:train_end]],
            'val': [samples[i] for i in indices[train_end:val_end]],
            'test': [samples[i] for i in indices[val_end:]]
        }
        
        print(f"\n=== Dataset Splits ===")
        for split_name, split_samples in splits.items():
            print(f"{split_name}: {len(split_samples)} samples")
            
            if format in ['json', 'both']:
                json_path = self.output_dir / f"stroke_letter_{split_name}.json"
                with open(json_path, 'w') as f:
                    json.dump(split_samples, f)
                print(f"  Saved: {json_path}")
            
            if format in ['pickle', 'both']:
                pkl_path = self.output_dir / f"stroke_letter_{split_name}.pkl"
                with open(pkl_path, 'wb') as f:
                    pickle.dump(split_samples, f)
                print(f"  Saved: {pkl_path}")
        
        # Save vocabulary (all unique letters)
        all_letters = set(s['letter'] for s in samples)
        vocab = {letter: idx for idx, letter in enumerate(sorted(all_letters))}
        
        vocab_path = self.output_dir / "vocabulary.json"
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"\nVocabulary ({len(vocab)} classes) saved: {vocab_path}")
        
        # Save dataset info
        info = {
            'total_samples': len(samples),
            'train_samples': len(splits['train']),
            'val_samples': len(splits['val']),
            'test_samples': len(splits['test']),
            'num_classes': len(vocab),
            'vocabulary': vocab,
            'letter_distribution': dict(sorted(
                [(l, sum(1 for s in samples if s['letter'] == l)) for l in vocab],
                key=lambda x: -x[1]
            ))
        }
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Dataset info saved: {info_path}")
        
        return splits


# ============================================================
# PyTorch Dataset (optional, for training)
# ============================================================

def create_pytorch_dataset():
    """
    Example PyTorch Dataset class for the stroke-letter data.
    Uncomment and use if you have PyTorch installed.
    """
    code = '''
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class StrokeLetterDataset(Dataset):
    """PyTorch Dataset for stroke -> letter classification"""
    
    def __init__(self, json_path: str, vocab_path: str, max_len: int = 100):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.max_len = max_len
        self.num_classes = len(self.vocab)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get normalized stroke points
        points = np.array(sample['stroke_points_normalized'], dtype=np.float32)
        
        # Pad or truncate to max_len
        if len(points) > self.max_len:
            points = points[:self.max_len]
        elif len(points) < self.max_len:
            padding = np.zeros((self.max_len - len(points), 3), dtype=np.float32)
            points = np.vstack([points, padding])
        
        # Get label
        letter = sample['letter']
        label = self.vocab.get(letter, 0)
        
        return {
            'points': torch.tensor(points),  # (max_len, 3)
            'label': torch.tensor(label, dtype=torch.long),
            'length': min(len(sample['stroke_points']), self.max_len),
            'letter': letter
        }

# Usage:
# train_dataset = StrokeLetterDataset('stroke_letter_train.json', 'vocabulary.json')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
'''
    return code


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Stroke → Letter Dataset')
    parser.add_argument('--data_root', type=str, 
                        default=r'D:\Projects\Personal\IR\data\words.tgz',
                        help='Path to IAM-OnDB data root')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\Projects\Personal\IR\data\stroke_letter_dataset',
                        help='Output directory for dataset')
    parser.add_argument('--max_lines', type=int, default=None,
                        help='Maximum number of lines to process (None for all)')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                        help='Minimum confidence for samples')
    parser.add_argument('--min_points', type=int, default=3,
                        help='Minimum points per stroke')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stroke -> Letter Dataset Builder")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max lines: {args.max_lines or 'All'}")
    print("=" * 60)
    
    # Build dataset
    builder = StrokeLetterDatasetBuilder(args.data_root, args.output_dir)
    samples = builder.build_dataset(
        max_lines=args.max_lines,
        min_confidence=args.min_confidence,
        min_points=args.min_points
    )
    
    if samples:
        # Save dataset
        builder.save_dataset(samples, format='both')
        
        # Show sample
        print("\n=== Sample Entry ===")
        sample = samples[0]
        print(f"Stroke ID: {sample['stroke_id']}")
        print(f"Letter: '{sample['letter']}'")
        print(f"Num points: {sample['num_points']}")
        print(f"Confidence: {sample['confidence']:.2f}")
        print(f"First 5 points: {sample['stroke_points'][:5]}")
        
        # Save PyTorch dataset code
        pytorch_code = create_pytorch_dataset()
        pytorch_path = Path(args.output_dir) / "pytorch_dataset.py"
        with open(pytorch_path, 'w') as f:
            f.write(pytorch_code)
        print(f"\nPyTorch Dataset code saved: {pytorch_path}")
    else:
        print("No samples generated!")


if __name__ == "__main__":
    main()

