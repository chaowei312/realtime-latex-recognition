"""
PyTorch Dataset for Line-Level Image-to-Text Recognition

Loads line images and transcriptions for decoder-only training.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image


class LineTextDataset(Dataset):
    """
    Dataset for line image → text recognition.
    
    Each sample contains:
    - image: Grayscale line image
    - text_ids: Tokenized text (with BOS)
    - target_ids: Target text (shifted, with EOS)
    """
    
    def __init__(
        self,
        data_path: str,
        vocab_path: str,
        image_height: int = 64,
        image_width: int = 256,
        max_text_len: int = 128,
        augment: bool = False,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.max_text_len = max_text_len
        self.augment = augment
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Special tokens
        self.pad_id = self.vocab.get('<PAD>', 0)
        self.bos_id = self.vocab.get('<BOS>', 1)
        self.eos_id = self.vocab.get('<EOS>', 2)
        self.unk_id = self.vocab.get('<UNK>', 3)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.vocab.get(c, self.unk_id) for c in text]
    
    def detokenize(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        chars = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in [self.pad_id, self.bos_id]:
                continue
            chars.append(self.idx_to_char.get(idx, '?'))
        return ''.join(chars)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        # Normalize path: replace backslashes and prepend data directory
        image_path = image_path.replace('\\', '/')
        if not Path(image_path).is_absolute():
            image_path = Path(__file__).parent / image_path
        
        try:
            img = Image.open(image_path).convert('L')  # Grayscale
        except Exception as e:
            # Return blank image on error
            print(f"Warning: Failed to load image {image_path}: {e}")
            return np.ones((self.image_height, self.image_width), dtype=np.float32)
        
        # Resize maintaining aspect ratio
        w, h = img.size
        new_h = self.image_height
        new_w = int(w * (new_h / h))
        
        if new_w > self.image_width:
            new_w = self.image_width
            new_h = int(h * (new_w / w))
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Create padded image (white background)
        padded = Image.new('L', (self.image_width, self.image_height), 255)
        
        # Center the image
        x_offset = (self.image_width - new_w) // 2
        y_offset = (self.image_height - new_h) // 2
        padded.paste(img, (x_offset, y_offset))
        
        # Convert to numpy and normalize to [0, 1]
        img_array = np.array(padded, dtype=np.float32) / 255.0
        
        return img_array
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Load image
        image = self.load_image(sample['image_path'])
        
        if self.augment:
            image = self._augment_image(image)
        
        # Tokenize text
        text = sample['text']
        token_ids = self.tokenize(text)
        
        # Truncate if needed
        if len(token_ids) > self.max_text_len - 2:  # Leave room for BOS and EOS
            token_ids = token_ids[:self.max_text_len - 2]
        
        # Create input (BOS + tokens) and target (tokens + EOS)
        input_ids = [self.bos_id] + token_ids
        target_ids = token_ids + [self.eos_id]
        
        # Pad to max length
        input_len = len(input_ids)
        pad_len = self.max_text_len - input_len
        
        input_ids = input_ids + [self.pad_id] * pad_len
        target_ids = target_ids + [self.pad_id] * pad_len
        
        # Create mask (1 for real tokens, 0 for padding)
        mask = [1] * input_len + [0] * pad_len
        
        return {
            'image': torch.from_numpy(image).unsqueeze(0),  # (1, H, W)
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'text': text,
            'length': input_len,
        }
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive data augmentation for SOTA training."""
        import random
        import cv2
        
        h, w = image.shape
        
        # Random rotation (-3° to +3°)
        if random.random() < 0.3:
            angle = random.uniform(-3, 3)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderValue=1.0)
        
        # Random shear
        if random.random() < 0.2:
            shear = random.uniform(-0.05, 0.05)
            M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
            image = cv2.warpAffine(image, M, (w, h), borderValue=1.0)
        
        # Random scale (slight zoom in/out)
        if random.random() < 0.2:
            scale = random.uniform(0.95, 1.05)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            # Crop or pad back to original size
            if scale > 1:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = image[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                padded = np.ones((h, w), dtype=np.float32)
                padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = image
                image = padded
        
        # Random erosion/dilation (pen thickness variation)
        if random.random() < 0.15:
            kernel = np.ones((2, 2), np.uint8)
            # Convert to uint8 for morphological ops
            img_uint8 = (image * 255).astype(np.uint8)
            if random.random() < 0.5:
                img_uint8 = cv2.erode(img_uint8, kernel, iterations=1)
            else:
                img_uint8 = cv2.dilate(img_uint8, kernel, iterations=1)
            image = img_uint8.astype(np.float32) / 255.0
        
        # Random Gaussian blur
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(0.01, 0.03), image.shape)
            image = np.clip(image + noise, 0, 1).astype(np.float32)
        
        # Random brightness
        if random.random() < 0.3:
            brightness = random.uniform(-0.1, 0.1)
            image = np.clip(image + brightness, 0, 1).astype(np.float32)
        
        # Random contrast
        if random.random() < 0.3:
            contrast = random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * contrast + mean, 0, 1).astype(np.float32)
        
        return image.astype(np.float32)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'masks': torch.stack([item['mask'] for item in batch]),
        'texts': [item['text'] for item in batch],
        'lengths': torch.tensor([item['length'] for item in batch]),
    }


def create_line_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_height: int = 64,
    image_width: int = 256,
    max_text_len: int = 128,
    num_workers: int = 0,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    data_dir = Path(data_dir)
    vocab_path = data_dir / 'vocabulary.json'
    
    train_dataset = LineTextDataset(
        data_path=data_dir / 'train.json',
        vocab_path=vocab_path,
        image_height=image_height,
        image_width=image_width,
        max_text_len=max_text_len,
        augment=augment_train,
    )
    
    val_dataset = LineTextDataset(
        data_path=data_dir / 'val.json',
        vocab_path=vocab_path,
        image_height=image_height,
        image_width=image_width,
        max_text_len=max_text_len,
        augment=False,
    )
    
    test_dataset = LineTextDataset(
        data_path=data_dir / 'test.json',
        vocab_path=vocab_path,
        image_height=image_height,
        image_width=image_width,
        max_text_len=max_text_len,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset (requires built dataset)
    import sys
    
    data_dir = Path(__file__).parent / 'line_text_dataset'
    
    if not data_dir.exists():
        print(f"Dataset not found at {data_dir}")
        print("Run build_line_dataset.py first!")
        sys.exit(1)
    
    print("Testing LineTextDataset...")
    
    dataset = LineTextDataset(
        data_path=data_dir / 'train.json',
        vocab_path=data_dir / 'vocabulary.json',
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Target IDs shape: {sample['target_ids'].shape}")
    print(f"  Text: {sample['text']}")
    print(f"  Length: {sample['length']}")
    
    # Test dataloader
    train_loader, _, _ = create_line_dataloaders(str(data_dir), batch_size=4)
    batch = next(iter(train_loader))
    
    print(f"\nBatch:")
    print(f"  Images: {batch['images'].shape}")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Texts: {batch['texts'][:2]}")
    
    print("\nAll tests passed!")

