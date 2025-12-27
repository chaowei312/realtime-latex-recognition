"""
Training Script for Context Tracker

Usage:
    python -m context_tracker.training.train --data_dir path/to/mathwriting
    
    # Or use default path (auto-detected):
    python -m context_tracker.training.train
    
    # With custom config:
    python -m context_tracker.training.train --batch_size 16 --lr 5e-5 --epochs 10
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from context_tracker.model import ContextTrackerConfig, ContextTrackerModel
from context_tracker.training import (
    EditDataset, SimpleTokenizer, collate_edit_batch,
    TrainingConfig, train_step, evaluate
)
from context_tracker.training.cached_dataset import (
    CachedEditDataset, collate_cached_batch
)
from context_tracker.data import get_mathwriting_path, MathWritingAtomic


# =============================================================================
# Logging Setup
# =============================================================================

class TrainingLogger:
    """Logger that saves per-head losses to file and console."""
    
    def __init__(self, log_dir: Path, experiment_name: str = "run"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}_{timestamp}"
        
        # Log file paths
        self.log_file = self.log_dir / f"{self.experiment_name}.log"
        self.metrics_file = self.log_dir / f"{self.experiment_name}_metrics.jsonl"
        
        # Setup file logger (use unique name to avoid duplicate handlers)
        self.logger = logging.getLogger(f"{self.experiment_name}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler with explicit stdout flush
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Metrics accumulator
        self.metrics_history = []
    
    def info(self, msg: str):
        self.logger.info(msg)
        # Flush handlers to ensure logs are written immediately
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_step(
        self, 
        step: int, 
        epoch: int,
        losses: Dict[str, float],
        lr: float,
        time_per_step: float
    ):
        """Log a training step with all head losses."""
        # Extract losses
        total_loss = losses.get('loss', 0)
        symbol_loss = losses.get('symbol_loss', 0)
        position_loss = losses.get('position_loss', 0)
        stroke_loss = losses.get('stroke_loss', 0)
        
        # Format message
        msg = (
            f"Step {step:6d} | Epoch {epoch:2d} | "
            f"Loss: {total_loss:7.4f} | "
            f"AR: {symbol_loss:6.4f} | "
            f"TPM: {position_loss:6.4f} | "
            f"SMM: {stroke_loss:6.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {time_per_step:.2f}s"
        )
        self.logger.info(msg)
        
        # Save to metrics file
        metrics = {
            'step': step,
            'epoch': epoch,
            'total_loss': float(total_loss) if torch.is_tensor(total_loss) else total_loss,
            'symbol_loss_AR': float(symbol_loss) if torch.is_tensor(symbol_loss) else symbol_loss,
            'position_loss_TPM': float(position_loss) if torch.is_tensor(position_loss) else position_loss,
            'stroke_loss_SMM': float(stroke_loss) if torch.is_tensor(stroke_loss) else stroke_loss,
            'learning_rate': lr,
            'time_per_step': time_per_step,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.metrics_history.append(metrics)
        
        # Flush handlers
        for handler in self.logger.handlers:
            handler.flush()
    
    def log_eval(self, step: int, epoch: int, metrics: Dict[str, float]):
        """Log evaluation metrics with accuracy/F1."""
        msg = (
            f"EVAL Step {step:6d} | Epoch {epoch:2d} | "
            f"Val Loss: {metrics.get('loss', 0):7.4f} | "
            f"Val AR: {metrics.get('symbol_loss', 0):6.4f} | "
            f"Val TPM: {metrics.get('position_loss', 0):6.4f} | "
            f"Val SMM: {metrics.get('stroke_loss', 0):6.4f}"
        )
        
        # Additional metrics line for accuracy/F1
        msg_metrics = (
            f"       TPM Acc: {metrics.get('tpm_acc', 0)*100:5.1f}% | "
            f"SMM P/R/F1: {metrics.get('smm_precision', 0)*100:4.1f}/"
            f"{metrics.get('smm_recall', 0)*100:4.1f}/"
            f"{metrics.get('smm_f1', 0)*100:4.1f}%"
        )
        
        self.logger.info("=" * 80)
        self.logger.info(msg)
        self.logger.info(msg_metrics)
        self.logger.info("=" * 80)
        
        # Also save to metrics file
        eval_metrics = {
            'step': step,
            'epoch': epoch,
            'type': 'eval',
            'val_loss': float(metrics.get('loss', 0)),
            'val_symbol_loss': float(metrics.get('symbol_loss', 0)),
            'val_position_loss': float(metrics.get('position_loss', 0)),
            'val_stroke_loss': float(metrics.get('stroke_loss', 0)),
            'tpm_accuracy': float(metrics.get('tpm_acc', 0)),
            'smm_precision': float(metrics.get('smm_precision', 0)),
            'smm_recall': float(metrics.get('smm_recall', 0)),
            'smm_f1': float(metrics.get('smm_f1', 0)),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(eval_metrics) + '\n')
    
    def log_config(self, model_config: ContextTrackerConfig, train_config: TrainingConfig):
        """Log configuration at start of training."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Model Config: {asdict(model_config)}")
        self.logger.info(f"Train Config: {asdict(train_config)}")
        self.logger.info("=" * 80)


# =============================================================================
# Training Loop
# =============================================================================

def train(
    model: ContextTrackerModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: TrainingConfig,
    model_config: ContextTrackerConfig,
    logger: TrainingLogger,
    resume_checkpoint: Optional[str] = None
):
    """
    Full training loop with logging and checkpointing.
    
    Logs per-head losses:
    - AR Loss (symbol_loss): Symbol prediction head
    - TPM Loss (position_loss): Tree-aware position module
    - SMM Loss (stroke_loss): Stroke modification module
    """
    device = torch.device(config.device)
    model = model.to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.warmup_steps,
        T_mult=2
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == 'cuda' else None
    
    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resumed from checkpoint: {resume_checkpoint} (step {start_step})")
    
    # Log config
    logger.log_config(model_config, config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Device: {device}")
    
    # Training loop
    global_step = start_step
    model.train()
    
    steps_per_epoch = len(train_loader)
    total_epochs = (config.max_steps - start_step) // steps_per_epoch + 1
    
    logger.info(f"Starting training for {config.max_steps - start_step} steps ({total_epochs} epochs)")
    logger.info("-" * 80)
    
    for epoch in range(start_epoch, start_epoch + total_epochs):
        epoch_start = time.time()
        epoch_losses = {'loss': 0, 'symbol_loss': 0, 'position_loss': 0, 'stroke_loss': 0}
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= config.max_steps:
                break
            
            step_start = time.time()
            
            # Forward + backward
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    losses = train_step(model, batch, config, global_step)
                scaler.scale(losses['loss']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = train_step(model, batch, config, global_step)
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            global_step += 1
            epoch_steps += 1
            
            # Accumulate epoch losses
            for key in epoch_losses:
                val = losses.get(key, 0)
                epoch_losses[key] += float(val) if torch.is_tensor(val) else val
            
            # Log every N steps
            if global_step % config.log_interval == 0:
                time_per_step = time.time() - step_start
                lr = scheduler.get_last_lr()[0]
                logger.log_step(global_step, epoch, losses, lr, time_per_step)
            
            # Evaluate every N steps
            if val_loader and global_step % config.eval_interval == 0:
                model.eval()
                val_metrics = evaluate(model, val_loader, config, max_batches=50)
                logger.log_eval(global_step, epoch, val_metrics)
                model.train()
            
            # Save checkpoint every N steps
            if global_step % config.save_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, epoch,
                    logger.log_dir / f"checkpoint_step{global_step}.pt"
                )
                logger.info(f"Saved checkpoint at step {global_step}")
        
        # End of epoch summary
        if epoch_steps > 0:
            avg_losses = {k: v / epoch_steps for k, v in epoch_losses.items()}
            epoch_time = time.time() - epoch_start
            logger.info("-" * 80)
            logger.info(
                f"Epoch {epoch} complete | "
                f"Avg Loss: {avg_losses['loss']:.4f} | "
                f"Avg AR: {avg_losses['symbol_loss']:.4f} | "
                f"Avg TPM: {avg_losses['position_loss']:.4f} | "
                f"Avg SMM: {avg_losses['stroke_loss']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            logger.info("-" * 80)
        
        if global_step >= config.max_steps:
            break
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, global_step, epoch,
        logger.log_dir / "checkpoint_final.pt"
    )
    logger.info(f"Training complete! Final checkpoint saved.")
    
    return model


def save_checkpoint(model, optimizer, step, epoch, path):
    """Save training checkpoint."""
    torch.save({
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Context Tracker")
    
    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to MathWriting data (auto-detected if not specified)')
    parser.add_argument('--cached_data', type=str, default=None,
                        help='Path to pre-computed cached data (from prepare_data.py)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit training samples (for testing)')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=128)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    # Loss weights
    parser.add_argument('--lambda_position', type=float, default=1.0,
                        help='Weight for TPM (position) loss')
    parser.add_argument('--lambda_stroke', type=float, default=0.5,
                        help='Weight for SMM (stroke) loss')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--experiment_name', type=str, default='context_tracker')
    
    # Misc
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable automatic mixed precision (AMP)')
    
    args = parser.parse_args()
    
    # Note: pin_memory=True can block on Windows with num_workers=0
    use_pin_memory = args.num_workers > 0
    
    # Check if using cached data
    if args.cached_data:
        # Use pre-computed cached data (FAST)
        print(f"Using CACHED data from: {args.cached_data}")
        
        train_dataset = CachedEditDataset(
            args.cached_data,
            split='synthetic',
            max_samples=args.max_samples
        )
        
        val_dataset = CachedEditDataset(
            args.cached_data,
            split='valid',
            max_samples=min(1000, args.max_samples) if args.max_samples else 1000
        )
        
        vocab_size = train_dataset.vocab_size
        collate_fn = collate_cached_batch
        
    else:
        # On-the-fly data generation (SLOW but flexible)
        if args.data_dir:
            data_path = Path(args.data_dir)
        else:
            data_path = get_mathwriting_path()
        
        print(f"Using MathWriting data from: {data_path}")
        print("(TIP: Use --cached_data for faster training after running prepare_data.py)")
        
        # Create tokenizer
        tokenizer = SimpleTokenizer()
        vocab_size = len(tokenizer)
        
        # Load datasets
        print("Loading MathWriting dataset...")
        train_mw = MathWritingAtomic(
            data_path, 
            split='synthetic', 
            lazy_load=True, 
            load_bboxes=True,
            max_samples=args.max_samples
        )
        
        val_mw = MathWritingAtomic(
            data_path,
            split='valid',
            lazy_load=True,
            load_bboxes=True,
            max_samples=min(1000, args.max_samples) if args.max_samples else 1000
        )
        
        print(f"Train samples: {len(train_mw)}")
        print(f"Val samples: {len(val_mw)}")
        
        # Create datasets
        train_dataset = EditDataset(
            train_mw, tokenizer,
            image_size=args.image_size,
            max_context_len=64,
            min_chunks=2,
            max_chunks=3
        )
        
        val_dataset = EditDataset(
            val_mw, tokenizer,
            image_size=args.image_size,
            max_context_len=64,
            min_chunks=2,
            max_chunks=3
        )
        
        collate_fn = collate_edit_batch
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model config
    model_config = ContextTrackerConfig(
        image_size=args.image_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=vocab_size,
        max_seq_len=512
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    
    # Create model
    print("Creating model...")
    model = ContextTrackerModel(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    steps_per_epoch = len(train_loader)
    max_steps = args.max_steps or (args.epochs * steps_per_epoch)
    
    train_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_steps=max_steps,
        warmup_steps=args.warmup_steps,
        lambda_position=args.lambda_position,
        lambda_stroke=args.lambda_stroke,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.log_dir,
        device=args.device,
        use_amp=args.use_amp
    )
    
    # Logger
    logger = TrainingLogger(Path(args.log_dir), args.experiment_name)
    
    # Train!
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_config=model_config,
        logger=logger,
        resume_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()

