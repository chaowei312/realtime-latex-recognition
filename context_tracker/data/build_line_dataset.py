"""
Build Line-Level Dataset for Image-to-Text Recognition

Creates (line_image, transcription) pairs from IAM-OnDB:
- lineImages-all: TIF images of handwritten lines
- ascii-all: Transcriptions (CSR = corrected version)

Output:
- line_text_dataset/
  - train.json, val.json, test.json
  - vocabulary.json
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm


def parse_ascii_file(ascii_path: Path) -> List[str]:
    """
    Parse ASCII transcription file.
    
    Format:
        OCR:
        <original text lines>
        
        CSR:
        <corrected text lines>  <- We use these
    
    Returns:
        List of corrected text lines (CSR section)
    """
    with open(ascii_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Find CSR section
    if 'CSR:' in content:
        csr_section = content.split('CSR:')[1].strip()
    else:
        # Fallback to OCR if no CSR
        if 'OCR:' in content:
            csr_section = content.split('OCR:')[1].strip()
            if 'CSR:' in csr_section:
                csr_section = csr_section.split('CSR:')[0].strip()
        else:
            csr_section = content.strip()
    
    # Split into lines
    lines = [line.strip() for line in csr_section.split('\n') if line.strip()]
    
    return lines


def find_line_images(form_dir: Path) -> Dict[str, Path]:
    """
    Find all line images in a form directory.
    
    Returns:
        Dict mapping line_id (e.g., 'a01-000u-01') to image path
    """
    images = {}
    
    for img_path in form_dir.glob('*.tif'):
        # Extract line ID from filename: a01-000u-01.tif -> a01-000u-01
        line_id = img_path.stem
        images[line_id] = img_path
    
    return images


def match_lines_to_images(
    ascii_path: Path,
    form_images_dir: Path,
) -> List[Dict]:
    """
    Match transcription lines to their corresponding images.
    
    Args:
        ascii_path: Path to .txt transcription file
        form_images_dir: Directory containing line images for this form
        
    Returns:
        List of {image_path, text, line_id} dicts
    """
    # Get form ID from ascii path: a01-000u.txt -> a01-000u
    form_id = ascii_path.stem
    
    # Parse transcriptions
    lines = parse_ascii_file(ascii_path)
    
    # Find images
    images = find_line_images(form_images_dir)
    
    # Match by line number
    matched = []
    for i, text in enumerate(lines):
        # Line ID format: form_id-XX where XX is line number (01, 02, ...)
        line_num = f"{i + 1:02d}"
        line_id = f"{form_id}-{line_num}"
        
        if line_id in images:
            matched.append({
                'image_path': str(images[line_id]),
                'text': text,
                'line_id': line_id,
                'form_id': form_id,
            })
    
    return matched


def build_vocabulary(samples: List[Dict], min_freq: int = 1) -> Dict[str, int]:
    """
    Build character-level vocabulary from samples.
    
    Special tokens:
        0: <PAD>
        1: <BOS>
        2: <EOS>
        3: <UNK>
    """
    # Count characters
    char_counts = Counter()
    for sample in samples:
        char_counts.update(sample['text'])
    
    # Build vocab
    vocab = {
        '<PAD>': 0,
        '<BOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    
    idx = 4
    for char, count in sorted(char_counts.items()):
        if count >= min_freq:
            vocab[char] = idx
            idx += 1
    
    return vocab


def build_dataset(
    data_root: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Build the complete line-level dataset.
    
    Args:
        data_root: Path to words.tgz directory
        output_dir: Output directory for dataset
        max_samples: Maximum samples to process (None = all)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    """
    random.seed(seed)
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ascii_root = data_root / 'ascii-all' / 'ascii'
    images_root = data_root / 'lineImages-all' / 'lineImages'
    
    print("=" * 60)
    print("Building Line-Level Dataset")
    print("=" * 60)
    print(f"ASCII root: {ascii_root}")
    print(f"Images root: {images_root}")
    print()
    
    # Collect all samples
    all_samples = []
    
    # Iterate through all form directories
    for set_dir in sorted(ascii_root.iterdir()):
        if not set_dir.is_dir():
            continue
        
        set_name = set_dir.name  # e.g., 'a01'
        images_set_dir = images_root / set_name
        
        if not images_set_dir.exists():
            continue
        
        for form_dir in sorted(set_dir.iterdir()):
            if not form_dir.is_dir():
                continue
            
            form_name = form_dir.name  # e.g., 'a01-000'
            images_form_dir = images_set_dir / form_name
            
            if not images_form_dir.exists():
                continue
            
            # Process each ASCII file in the form directory
            for ascii_file in form_dir.glob('*.txt'):
                matched = match_lines_to_images(ascii_file, images_form_dir)
                all_samples.extend(matched)
        
        if max_samples and len(all_samples) >= max_samples:
            all_samples = all_samples[:max_samples]
            break
    
    print(f"Total samples found: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("ERROR: No samples found!")
        return
    
    # Build vocabulary
    vocab = build_vocabulary(all_samples)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Shuffle and split
    random.shuffle(all_samples)
    
    n_train = int(len(all_samples) * train_ratio)
    n_val = int(len(all_samples) * val_ratio)
    
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]
    
    print(f"Train: {len(train_samples)}")
    print(f"Val: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")
    
    # Compute statistics
    text_lengths = [len(s['text']) for s in all_samples]
    avg_len = sum(text_lengths) / len(text_lengths)
    max_len = max(text_lengths)
    
    print(f"Avg text length: {avg_len:.1f} chars")
    print(f"Max text length: {max_len} chars")
    
    # Save
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(output_dir / 'val.json', 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    with open(output_dir / 'test.json', 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    with open(output_dir / 'vocabulary.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Save dataset info
    info = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'vocab_size': len(vocab),
        'avg_text_length': avg_len,
        'max_text_length': max_len,
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print()
    print(f"Dataset saved to: {output_dir}")
    print("=" * 60)
    
    # Print sample
    print("\nSample entry:")
    print(json.dumps(train_samples[0], indent=2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='words.tgz',
                        help='Path to words.tgz directory')
    parser.add_argument('--output_dir', type=str,
                        default='line_text_dataset',
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples (None = all)')
    
    args = parser.parse_args()
    
    build_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

