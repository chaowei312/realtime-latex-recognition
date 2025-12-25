"""
Test Trajectory-Based Autoregressive Text Recognition

Given stroke trajectories of a short sentence, the model should autoregressively
decode the text. Uses trajectory-based patch extraction instead of full image scan.

Architecture:
1. Stroke trajectories → TrajectoryPatchExtractor → Patches along trajectory
2. Patches → StrokePatchEncoder → Patch embeddings
3. [Patch embeddings] [CLS] → Decoder → Autoregressive text output

Usage:
    python test_trajectory_ar.py --epochs 20 --encoder-variant stacked_3x3_s2
"""

import os
import sys
import json
import time
import math
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from context_tracker.model.module import (
    StrokePatchEncoder,
    TrajectoryPatchExtractor,
    list_available_variants,
    SwiGLUFFN,
)


# ============================================================================
# Dataset: Line-level stroke sequences with text labels
# ============================================================================

class StrokeLineDataset(Dataset):
    """
    Dataset for line-level stroke recognition.
    
    Each sample contains:
    - Multiple strokes forming a text line (or word)
    - Ground truth text
    
    We simulate this by grouping individual stroke-letter samples.
    """
    
    def __init__(
        self,
        stroke_data_path: str,
        line_data_path: str,
        vocab_path: str,
        max_strokes: int = 20,
        max_text_len: int = 50,
        canvas_size: Tuple[int, int] = (128, 512),
        patch_size: int = 16,
        sample_interval: int = 8,
    ):
        self.max_strokes = max_strokes
        self.max_text_len = max_text_len
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.sample_interval = sample_interval
        
        # Load line data (for text targets)
        with open(line_data_path, 'r') as f:
            self.line_data = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Special tokens
        self.pad_id = self.vocab.get('<PAD>', 0)
        self.bos_id = self.vocab.get('<BOS>', 1)
        self.eos_id = self.vocab.get('<EOS>', 2)
        
        # Try to load stroke data (optional - will synthesize if not available)
        self.stroke_data = None
        if stroke_data_path and Path(stroke_data_path).exists():
            with open(stroke_data_path, 'r') as f:
                self.stroke_data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.line_data)
    
    def __getitem__(self, idx: int) -> Dict:
        line = self.line_data[idx]
        text = line['text']
        
        # Generate synthetic strokes for this text
        strokes = self._generate_strokes_for_text(text)
        
        # Render strokes to canvas
        canvas = self._render_strokes_to_canvas(strokes)
        
        # Extract trajectory patches
        all_patches = []
        stroke_ids = []
        
        for stroke_idx, stroke in enumerate(strokes):
            patches = self._extract_trajectory_patches(stroke, canvas)
            all_patches.extend(patches)
            stroke_ids.extend([stroke_idx] * len(patches))
        
        # Convert to tensors
        if all_patches:
            patches_tensor = torch.stack(all_patches)
        else:
            patches_tensor = torch.zeros(1, 1, self.patch_size, self.patch_size)
        
        # Tokenize text
        input_ids, target_ids = self._tokenize_text(text)
        
        return {
            'patches': patches_tensor,  # (num_patches, 1, H, W)
            'stroke_ids': torch.tensor(stroke_ids, dtype=torch.long),
            'num_patches': len(all_patches),
            'num_strokes': len(strokes),
            'input_ids': input_ids,  # (max_len,) with BOS
            'target_ids': target_ids,  # (max_len,) with EOS
            'text': text,
            'text_len': len(text),
        }
    
    def _generate_strokes_for_text(self, text: str) -> List[List[Tuple[float, float, float]]]:
        """
        Generate synthetic stroke trajectories for text.
        
        Each character becomes a stroke with points along a path.
        """
        strokes = []
        x_offset = 20
        char_width = 20
        char_height = 40
        y_base = self.canvas_size[0] // 2
        
        for i, char in enumerate(text):
            if char == ' ':
                x_offset += char_width // 2
                continue
            
            # Generate stroke points for this character
            x_start = x_offset + i * char_width
            stroke = self._generate_char_stroke(char, x_start, y_base, char_width, char_height)
            strokes.append(stroke)
            
            if len(strokes) >= self.max_strokes:
                break
        
        return strokes
    
    def _generate_char_stroke(
        self, 
        char: str, 
        x: float, 
        y: float, 
        w: float, 
        h: float
    ) -> List[Tuple[float, float, float]]:
        """Generate a simple stroke pattern for a character."""
        points = []
        num_points = 20
        
        # Simple zigzag pattern (varies by character hash)
        seed = ord(char)
        np.random.seed(seed)
        
        for i in range(num_points):
            t = i / (num_points - 1)
            px = x + t * w + np.random.normal(0, 2)
            py = y + np.sin(t * np.pi * 2) * h * 0.3 + np.random.normal(0, 2)
            pt = i * 0.01  # Time
            points.append((px, py, pt))
        
        np.random.seed(None)  # Reset
        return points
    
    def _render_strokes_to_canvas(
        self, 
        strokes: List[List[Tuple[float, float, float]]]
    ) -> torch.Tensor:
        """Render all strokes to a canvas image."""
        H, W = self.canvas_size
        canvas = torch.ones(1, H, W)  # White background
        
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x0, y0, _ = stroke[i]
                x1, y1, _ = stroke[i + 1]
                
                # Draw line segment
                num_steps = max(int(abs(x1 - x0) + abs(y1 - y0)), 1)
                for t in range(num_steps + 1):
                    px = int(x0 + (x1 - x0) * t / num_steps)
                    py = int(y0 + (y1 - y0) * t / num_steps)
                    
                    if 0 <= px < W and 0 <= py < H:
                        # Draw with thickness
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = px + dx, py + dy
                                if 0 <= nx < W and 0 <= ny < H:
                                    canvas[0, ny, nx] = 0.0  # Black ink
        
        return canvas
    
    def _extract_trajectory_patches(
        self,
        stroke: List[Tuple[float, float, float]],
        canvas: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Extract patches along stroke trajectory."""
        patches = []
        half = self.patch_size // 2
        C, H, W = canvas.shape
        
        for i in range(0, len(stroke), self.sample_interval):
            x, y, _ = stroke[i]
            x, y = int(x), int(y)
            
            # Extract patch centered at (x, y)
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(W, x + half)
            y2 = min(H, y + half)
            
            patch = canvas[:, y1:y2, x1:x2]
            
            # Pad if necessary
            if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
                padded = torch.ones(C, self.patch_size, self.patch_size)
                ph, pw = patch.shape[1], patch.shape[2]
                padded[:, :ph, :pw] = patch
                patch = padded
            
            patches.append(patch)
        
        return patches
    
    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text to input/target sequences."""
        # Convert to indices
        indices = [self.vocab.get(c, self.vocab.get('<UNK>', 0)) for c in text]
        
        # Truncate if needed
        indices = indices[:self.max_text_len - 2]  # Leave room for BOS/EOS
        
        # Create input (with BOS) and target (with EOS)
        input_ids = [self.bos_id] + indices
        target_ids = indices + [self.eos_id]
        
        # Pad to max length
        input_ids = input_ids + [self.pad_id] * (self.max_text_len - len(input_ids))
        target_ids = target_ids + [self.pad_id] * (self.max_text_len - len(target_ids))
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def collate_stroke_lines(batch: List[Dict]) -> Dict:
    """Collate function for variable-length patch sequences."""
    # Find max patches
    max_patches = max(item['num_patches'] for item in batch)
    if max_patches == 0:
        max_patches = 1
    
    # Pad patches
    padded_patches = []
    patch_mask = []
    
    for item in batch:
        patches = item['patches']
        n = patches.shape[0]
        
        if n < max_patches:
            pad = torch.ones(max_patches - n, *patches.shape[1:])
            patches = torch.cat([patches, pad], dim=0)
            mask = [1] * n + [0] * (max_patches - n)
        else:
            mask = [1] * max_patches
        
        padded_patches.append(patches)
        patch_mask.append(mask)
    
    return {
        'patches': torch.stack(padded_patches),  # (B, max_patches, C, H, W)
        'patch_mask': torch.tensor(patch_mask, dtype=torch.bool),  # (B, max_patches)
        'num_patches': torch.tensor([item['num_patches'] for item in batch]),
        'num_strokes': torch.tensor([item['num_strokes'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'target_ids': torch.stack([item['target_ids'] for item in batch]),
        'texts': [item['text'] for item in batch],
        'text_lens': torch.tensor([item['text_len'] for item in batch]),
    }


# ============================================================================
# Model: Trajectory-based Encoder-Decoder
# ============================================================================

class TrajectoryARModel(nn.Module):
    """
    Trajectory-based Autoregressive Text Recognition Model.
    
    Architecture:
    1. Patches → StrokePatchEncoder → Patch embeddings
    2. Patch embeddings → MeanPool → [CLS] context
    3. [CLS] + Text embeddings → Decoder → Autoregressive output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        encoder_variant: str = "stacked_3x3_s2",
        pad_id: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_id = pad_id
        
        # Patch encoder
        self.patch_encoder = StrokePatchEncoder(
            variant=encoder_variant,
            embed_dim=embed_dim,
            input_channels=1,
            input_size=16,
        )
        
        # CLS token for aggregating patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Patch position embedding (learned)
        self.max_patches = 256
        self.patch_pos = nn.Parameter(torch.randn(1, self.max_patches, embed_dim) * 0.02)
        
        # Text embeddings
        self.text_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.text_pos = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        
        self.max_seq_len = max_seq_len
    
    def encode_patches(
        self, 
        patches: torch.Tensor, 
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode patch sequence to context embedding.
        
        Args:
            patches: (batch, num_patches, C, H, W)
            patch_mask: (batch, num_patches) - True for valid patches
            
        Returns:
            context: (batch, 1, embed_dim) - CLS representation
        """
        B, N, C, H, W = patches.shape
        
        # Flatten batch and patches
        patches_flat = patches.view(B * N, C, H, W)
        
        # Encode all patches
        patch_embeds = self.patch_encoder(patches_flat)  # (B*N, embed_dim)
        patch_embeds = patch_embeds.view(B, N, -1)  # (B, N, embed_dim)
        
        # Add position embeddings
        if N <= self.max_patches:
            patch_embeds = patch_embeds + self.patch_pos[:, :N, :]
        
        # Apply mask (set invalid patches to zero)
        if patch_mask is not None:
            patch_embeds = patch_embeds * patch_mask.unsqueeze(-1).float()
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        
        # CLS attends to all patches (mean pooling for simplicity)
        if patch_mask is not None:
            # Masked mean
            mask_sum = patch_mask.sum(dim=1, keepdim=True).clamp(min=1)
            context = (patch_embeds * patch_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)
        else:
            context = patch_embeds.mean(dim=1, keepdim=True)
        
        return context  # (B, 1, embed_dim)
    
    def decode(
        self,
        context: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode text autoregressively.
        
        Args:
            context: (batch, 1, embed_dim) - visual context
            input_ids: (batch, seq_len) - input token IDs
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, L = input_ids.shape
        
        # Embed text tokens
        text_embeds = self.text_embed(input_ids)  # (B, L, embed_dim)
        text_embeds = text_embeds + self.text_pos[:, :L, :]
        
        # Prepend visual context
        x = torch.cat([context, text_embeds], dim=1)  # (B, 1+L, embed_dim)
        
        # Create causal mask
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        
        # Decode
        for layer in self.decoder_layers:
            x = layer(x, causal_mask)
        
        x = self.out_norm(x)
        
        # Only output for text positions (skip context position)
        logits = self.out_proj(x[:, 1:, :])  # (B, L, vocab_size)
        
        return logits
    
    def forward(
        self,
        patches: torch.Tensor,
        input_ids: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        context = self.encode_patches(patches, patch_mask)
        logits = self.decode(context, input_ids)
        return {'logits': logits}
    
    @torch.no_grad()
    def generate(
        self,
        patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        bos_id: int = 1,
        eos_id: int = 2,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        B = patches.shape[0]
        device = patches.device
        
        # Encode patches
        context = self.encode_patches(patches, patch_mask)
        
        # Start with BOS
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            logits = self.decode(context, generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (generated == eos_id).any(dim=1).all():
                break
        
        return generated


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and FFN."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        patches = batch['patches'].to(device)
        patch_mask = batch['patch_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        optimizer.zero_grad()
        
        output = model(patches, input_ids, patch_mask)
        logits = output['logits']
        
        # Compute loss (ignore padding)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * target_ids.size(0)
        
        pred = logits.argmax(dim=-1)
        mask = target_ids != model.pad_id
        total_correct += ((pred == target_ids) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        pbar.set_postfix(loss=loss.item(), acc=total_correct/max(total_tokens, 1))
    
    return total_loss / len(dataloader.dataset), total_correct / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, idx_to_char, bos_id, eos_id, pad_id):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_cer = 0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        patches = batch['patches'].to(device)
        patch_mask = batch['patch_mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        targets = batch['texts']
        
        # Compute loss
        output = model(patches, input_ids, patch_mask)
        logits = output['logits']
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        total_loss += loss.item() * patches.size(0)
        
        # Generate predictions
        generated = model.generate(
            patches, patch_mask, 
            max_length=50,
            bos_id=bos_id,
            eos_id=eos_id,
            temperature=0.0,
        )
        
        # Compute CER
        for i in range(len(targets)):
            pred_tokens = generated[i].tolist()
            pred_str = decode_tokens(pred_tokens, idx_to_char, eos_id, bos_id, pad_id)
            target_str = targets[i]
            
            cer = compute_cer(pred_str, target_str)
            total_cer += cer
            num_samples += 1
    
    return total_loss / len(dataloader.dataset), total_cer / max(num_samples, 1)


def decode_tokens(token_ids, idx_to_char, eos_id, bos_id, pad_id):
    """Convert token IDs to string."""
    chars = []
    for idx in token_ids:
        if idx == eos_id:
            break
        if idx in [bos_id, pad_id]:
            continue
        chars.append(idx_to_char.get(idx, '?'))
    return ''.join(chars)


def compute_cer(pred: str, target: str) -> float:
    """Character Error Rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # Levenshtein distance
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / len(target)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Trajectory AR Model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--encoder-variant", type=str, default="stacked_3x3_s2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-dir", type=str, 
                        default=str(Path(__file__).parent / "data"))
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Encoder variant: {args.encoder_variant}")
    
    # Paths
    data_dir = Path(args.data_dir)
    line_dir = data_dir / "line_text_dataset"
    stroke_dir = data_dir / "stroke_letter_dataset"
    
    # Load vocabulary
    vocab_path = line_dir / "vocabulary.json"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    idx_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Special tokens
    pad_id = vocab.get('<PAD>', 0)
    bos_id = vocab.get('<BOS>', 1)
    eos_id = vocab.get('<EOS>', 2)
    
    # Create datasets
    train_dataset = StrokeLineDataset(
        stroke_data_path=str(stroke_dir / "stroke_letter_train.json") if stroke_dir.exists() else None,
        line_data_path=str(line_dir / "train.json"),
        vocab_path=str(vocab_path),
        max_strokes=20,
        max_text_len=80,
    )
    
    val_dataset = StrokeLineDataset(
        stroke_data_path=str(stroke_dir / "stroke_letter_val.json") if stroke_dir.exists() else None,
        line_data_path=str(line_dir / "val.json"),
        vocab_path=str(vocab_path),
        max_strokes=20,
        max_text_len=80,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_stroke_lines,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_stroke_lines,
        num_workers=0,
    )
    
    # Create model
    model = TrajectoryARModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=80,
        dropout=0.1,
        encoder_variant=args.encoder_variant,
        pad_id=pad_id,
    ).to(device)
    
    # Print model info
    encoder_config = model.patch_encoder.get_config_summary()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder: {encoder_config['name']} ({encoder_config['num_params']:,} params)")
    print(f"Total model params: {total_params:,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # Training loop
    best_cer = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_cer = evaluate(model, val_loader, criterion, device, idx_to_char, bos_id, eos_id, pad_id)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, CER={val_cer:.4f} | "
              f"Time={epoch_time:.1f}s")
        
        if val_cer < best_cer:
            best_cer = val_cer
            print(f"  → New best CER: {best_cer:.4f}")
    
    print(f"\nFinal best CER: {best_cer:.4f}")
    
    # Show sample predictions
    print("\nSample predictions:")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        patches = batch['patches'][:3].to(device)
        patch_mask = batch['patch_mask'][:3].to(device)
        targets = batch['texts'][:3]
        
        generated = model.generate(
            patches, patch_mask,
            max_length=50,
            bos_id=bos_id,
            eos_id=eos_id,
            temperature=0.0,
        )
        
        for i in range(min(3, len(targets))):
            pred = decode_tokens(generated[i].tolist(), idx_to_char, eos_id, bos_id, pad_id)
            print(f"  Target: '{targets[i]}'")
            print(f"  Pred:   '{pred}'")
            print()


if __name__ == "__main__":
    main()

