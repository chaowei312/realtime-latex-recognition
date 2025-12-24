"""
SOTA Training Script for Decoder-Only Image-to-Text Model

Features:
- Large model configuration
- Cosine annealing with warmup
- Label smoothing
- Gradient accumulation
- Comprehensive data augmentation
- CER/WER evaluation metrics
"""

import os
import sys
import math
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from model import DecoderOnlyImageToText, DecoderOnlyConfig
from data.line_dataset import LineTextDataset, create_line_dataloaders, collate_fn


# ============================================================================
# Label Smoothing Loss
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L, V) - model predictions
            targets: (B, L) - target token IDs
        """
        V = logits.size(-1)
        logits = logits.view(-1, V)
        targets = targets.view(-1)
        
        # Mask for valid tokens
        mask = targets != self.ignore_index
        targets_masked = targets.clone()
        targets_masked[~mask] = 0
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # NLL loss (hard targets)
        nll_loss = -log_probs.gather(dim=-1, index=targets_masked.unsqueeze(-1)).squeeze(-1)
        
        # Smooth loss (uniform distribution)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combined loss
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        loss = loss * mask.float()
        
        return loss.sum() / mask.sum().clamp(min=1)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_cer(pred: str, target: str) -> float:
    """Character Error Rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, target) / len(target)


def compute_wer(pred: str, target: str) -> float:
    """Word Error Rate."""
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein_distance(' '.join(pred_words), ' '.join(target_words)) / len(' '.join(target_words))


def decode_tokens(token_ids, idx_to_char, eos_id=2, bos_id=1, pad_id=0):
    """Convert token IDs to string."""
    chars = []
    for idx in token_ids:
        if idx == eos_id:
            break
        if idx in [bos_id, pad_id]:
            continue
        chars.append(idx_to_char.get(idx, '?'))
    return ''.join(chars)


# ============================================================================
# Training Utilities
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine annealing schedule with linear warmup."""
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model, dataloader, device, idx_to_char, config, max_samples=None):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    total_cer = 0
    total_wer = 0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        targets = batch['texts']
        
        # Compute loss
        output = model(images, input_ids)
        loss = criterion(
            output['logits'].view(-1, config.vocab_size),
            target_ids.view(-1)
        )
        total_loss += loss.item() * images.size(0)
        
        # Generate predictions (greedy decoding)
        generated = model.generate(images, max_length=128, temperature=0.0)
        
        for pred_ids, target in zip(generated, targets):
            pred_text = decode_tokens(
                pred_ids.cpu().tolist(), idx_to_char,
                eos_id=config.eos_token_id,
                bos_id=config.bos_token_id,
                pad_id=config.pad_token_id
            )
            total_cer += compute_cer(pred_text, target)
            total_wer += compute_wer(pred_text, target)
            total_samples += 1
        
        if max_samples and total_samples >= max_samples:
            break
    
    return {
        'loss': total_loss / total_samples,
        'cer': total_cer / total_samples,
        'wer': total_wer / total_samples,
    }


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, config, accum_steps=1):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        output = model(images, input_ids)
        loss = criterion(output['logits'], target_ids)
        loss = loss / accum_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accum_steps:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / num_batches


# ============================================================================
# Main
# ============================================================================

def main(args):
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    vocab_path = Path(args.data_dir) / 'vocabulary.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    vocab_size = len(vocab)
    idx_to_char = {v: k for k, v in vocab.items()}
    print(f"Vocabulary size: {vocab_size}")
    
    # Model configuration
    if args.model_size == 'small':
        config = DecoderOnlyConfig(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            ffn_dim=1024,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            max_visual_tokens=256,
            encoder_base_channels=32,
            encoder_num_stages=4,
        )
    elif args.model_size == 'base':
        config = DecoderOnlyConfig(
            vocab_size=vocab_size,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            ffn_dim=2048,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            max_visual_tokens=512,
            encoder_base_channels=64,
            encoder_num_stages=4,
        )
    elif args.model_size == 'large':
        config = DecoderOnlyConfig(
            vocab_size=vocab_size,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            ffn_dim=3072,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len,
            max_visual_tokens=512,
            encoder_base_channels=64,
            encoder_num_stages=5,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # Create model
    model = DecoderOnlyImageToText(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {args.model_size}")
    print(f"Model parameters: {num_params:,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_line_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_height=64,
        image_width=256,
        max_text_len=args.max_seq_len,
        num_workers=args.num_workers,
        augment_train=True,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.accum_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.accum_steps
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio=0.1
    )
    
    # Loss function
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        ignore_index=config.pad_token_id
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_cer = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
        best_cer = metrics.get('best_cer', float('inf'))
        start_epoch += 1
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    train_losses = []
    val_metrics_history = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, config, accum_steps=args.accum_steps
        )
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(model, val_loader, device, idx_to_char, config)
            val_metrics_history.append(val_metrics)
            
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val CER: {val_metrics['cer']:.4f} ({val_metrics['cer']*100:.2f}%)")
            print(f"Val WER: {val_metrics['wer']:.4f} ({val_metrics['wer']*100:.2f}%)")
            
            # Save best model
            if val_metrics['cer'] < best_cer:
                best_cer = val_metrics['cer']
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {'best_cer': best_cer, **val_metrics},
                    output_dir / 'best_model.pt'
                )
                print(f"New best CER: {best_cer:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train_loss': train_loss, 'best_cer': best_cer},
                output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, idx_to_char, config)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test CER: {test_metrics['cer']:.4f} ({test_metrics['cer']*100:.2f}%)")
    print(f"Test WER: {test_metrics['wer']:.4f} ({test_metrics['wer']*100:.2f}%)")
    
    # Save final results
    results = {
        'config': {
            'model_size': args.model_size,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'num_params': num_params,
        },
        'train_losses': train_losses,
        'val_metrics': val_metrics_history,
        'test_metrics': test_metrics,
        'best_cer': best_cer,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOTA Training for Image-to-Text')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/line_text_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/sota',
                        help='Output directory for checkpoints and logs')
    
    # Model
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='Model size configuration')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    
    # Misc
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)

