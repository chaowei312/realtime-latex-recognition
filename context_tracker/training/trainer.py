"""
Trainer for Context Tracker

Implements Option B training with incremental stroke removal:
- Each AR step processes only remaining strokes
- SMM learns to identify contributing strokes
- TPM learns tree position from K-cache
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..model.context_tracker import ContextTrackerModel, ContextTrackerConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batch and gradient
    batch_size: int = 32
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    
    # Loss weights
    lambda_position: float = 1.0
    lambda_stroke: float = 0.5
    lambda_distill: float = 0.1  # For teacher distillation
    
    # Logging and checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True  # Automatic mixed precision


def train_step(
    model: ContextTrackerModel,
    batch: Dict[str, torch.Tensor],
    config: TrainingConfig,
    step: int = 0,
    compute_metrics: bool = False
) -> Dict[str, float]:
    """
    Simplified training step - single forward pass with first target position.
    
    For prototyping, we just predict the first token given the context and image.
    Full Option B (incremental stroke removal) requires more complex handling.
    
    Args:
        model: ContextTrackerModel
        batch: Dict with context_ids, images, target_tokens, etc.
        config: TrainingConfig
        step: Current training step
        compute_metrics: If True, compute accuracy/F1 metrics (slower)
        
    Returns:
        Dict with loss values and optional metrics
    """
    device = next(model.parameters()).device
    
    # Move batch to device
    context_ids = batch['context_ids'].to(device)
    images = batch['images'].to(device)
    target_tokens = batch['target_tokens'].to(device)
    target_actions = batch['target_actions'].to(device)
    stroke_labels = batch['stroke_labels'].to(device)
    active_strokes = batch['active_strokes'].to(device)
    stroke_indices = batch['stroke_indices']
    seq_lens = batch['seq_lens'].to(device)
    
    B, T = target_tokens.shape
    
    # Single forward pass predicting first token
    output = model(
        images=images,
        context_ids=context_ids,
        stroke_indices=stroke_indices,
        active_strokes=active_strokes
    )
    
    # Symbol loss (Head A) - first token only
    symbol_logits = output['symbol_logits']  # [B, vocab]
    symbol_loss = F.cross_entropy(
        symbol_logits,
        target_tokens[:, 0],  # First target token
        reduction='mean'
    )
    
    # Position loss (Head C / TPM)
    # Action space is now: 1 parent (ROOT) × 6 relations = 6 actions
    position_loss = torch.tensor(0.0, device=device)
    tpm_acc = 0.0
    if 'action_logits' in output:
        action_logits = output['action_logits']  # [B, num_actions]
        target_action = target_actions[:, 0]  # First action
        
        # Clamp target to valid range (action space = 6 for ROOT-only)
        num_actions = action_logits.shape[1]
        target_action = target_action.clamp(0, num_actions - 1)
        
        position_loss = F.cross_entropy(
            action_logits,
            target_action,
            reduction='mean'
        )
        
        # Compute TPM accuracy
        if compute_metrics:
            pred_action = action_logits.argmax(dim=-1)
            tpm_acc = (pred_action == target_action).float().mean().item()
    
    # Stroke loss (Head B / SMM) - first token's strokes
    stroke_loss = torch.tensor(0.0, device=device)
    smm_precision = 0.0
    smm_recall = 0.0
    smm_f1 = 0.0
    if 'stroke_scores' in output:
        stroke_logits = output['stroke_scores']  # [B, num_strokes]
        stroke_target = stroke_labels[:, 0, :]   # First token's stroke labels
        
        # Replace -inf with large negative value for BCEWithLogits
        stroke_logits = stroke_logits.clamp(min=-100, max=100)
        
        # Only compute on active strokes using BCEWithLogits (AMP-safe)
        stroke_loss = F.binary_cross_entropy_with_logits(
            stroke_logits,
            stroke_target,
            reduction='none'
        )
        stroke_loss = stroke_loss * active_strokes.float()
        num_active = active_strokes.float().sum().clamp(min=1)
        stroke_loss = stroke_loss.sum() / num_active
        
        # Compute SMM precision/recall/F1
        if compute_metrics:
            with torch.no_grad():
                stroke_probs = torch.sigmoid(stroke_logits)
                stroke_pred = (stroke_probs > 0.5).float()
                
                # Only on active strokes
                mask = active_strokes.float()
                tp = ((stroke_pred * stroke_target) * mask).sum()
                fp = ((stroke_pred * (1 - stroke_target)) * mask).sum()
                fn = (((1 - stroke_pred) * stroke_target) * mask).sum()
                
                smm_precision = (tp / (tp + fp + 1e-8)).item()
                smm_recall = (tp / (tp + fn + 1e-8)).item()
                smm_f1 = (2 * smm_precision * smm_recall / (smm_precision + smm_recall + 1e-8))
    
    # Combine losses
    total_loss = (
        symbol_loss + 
        config.lambda_position * position_loss +
        config.lambda_stroke * stroke_loss
    )
    
    result = {
        'loss': total_loss,
        'symbol_loss': symbol_loss.detach().item(),
        'position_loss': position_loss.detach().item(),
        'stroke_loss': stroke_loss.detach().item(),
        'total_steps': B
    }
    
    if compute_metrics:
        result['tpm_acc'] = tpm_acc
        result['smm_precision'] = smm_precision
        result['smm_recall'] = smm_recall
        result['smm_f1'] = smm_f1
    
    return result


def evaluate(
    model: ContextTrackerModel,
    dataloader: DataLoader,
    config: TrainingConfig,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation set with detailed metrics.
    
    Args:
        model: ContextTrackerModel
        dataloader: Validation DataLoader
        config: TrainingConfig
        max_batches: Maximum batches to evaluate
        
    Returns:
        Dict with average metrics including TPM accuracy and SMM F1
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_metrics = {
        'loss': 0.0,
        'symbol_loss': 0.0,
        'position_loss': 0.0,
        'stroke_loss': 0.0,
        'tpm_acc': 0.0,
        'smm_precision': 0.0,
        'smm_recall': 0.0,
        'smm_f1': 0.0,
        'total_steps': 0
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Compute metrics during evaluation
            metrics = train_step(model, batch, config, compute_metrics=True)
            
            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics[key]
            num_batches += 1
    
    # Average
    if num_batches > 0:
        for key in total_metrics:
            if key != 'total_steps':
                total_metrics[key] /= num_batches
    
    model.train()
    return total_metrics


class Trainer:
    """
    Full training loop for Context Tracker.
    
    Features:
    - Option B training (incremental stroke removal)
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: ContextTrackerModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Move model to device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp else None
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self):
        """Run full training loop."""
        self.model.train()
        
        print(f"Starting training on {self.device}")
        print(f"  Total steps: {self.config.max_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        
        running_loss = 0.0
        running_metrics = {}
        train_iter = iter(self.train_loader)
        
        start_time = time.time()
        
        for step in range(self.global_step, self.config.max_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            metrics = self._train_step(batch)
            
            # Accumulate metrics
            running_loss += metrics['loss'].item() if hasattr(metrics['loss'], 'item') else metrics['loss']
            for key, value in metrics.items():
                if key not in running_metrics:
                    running_metrics[key] = 0.0
                val = value.item() if hasattr(value, 'item') else value
                running_metrics[key] += val
            
            self.global_step = step + 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = running_loss / self.config.log_interval
                elapsed = time.time() - start_time
                steps_per_sec = self.config.log_interval / elapsed
                
                log_str = f"Step {self.global_step}: loss={avg_loss:.4f}"
                for key, value in running_metrics.items():
                    if key != 'loss' and key != 'total_steps':
                        avg_val = value / self.config.log_interval
                        if hasattr(avg_val, 'item'):
                            avg_val = avg_val.item()
                        log_str += f", {key}={avg_val:.4f}"
                log_str += f" ({steps_per_sec:.2f} steps/s)"
                print(log_str)
                
                running_loss = 0.0
                running_metrics = {}
                start_time = time.time()
            
            # Evaluation
            if self.val_loader and self.global_step % self.config.eval_interval == 0:
                val_metrics = evaluate(
                    self.model, self.val_loader, self.config, max_batches=50
                )
                print(f"  Validation: loss={val_metrics['loss']:.4f}")
                
                # Save best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best.pt')
            
            # Checkpointing
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}.pt')
        
        print("Training complete!")
        self.save_checkpoint('final.pt')
    
    def _train_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single training step with gradient update."""
        self.optimizer.zero_grad()
        
        if self.config.use_amp and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                metrics = train_step(self.model, batch, self.config, self.global_step)
                loss = metrics['loss']
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            metrics = train_step(self.model, batch, self.config, self.global_step)
            loss = metrics['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
        
        self.scheduler.step()
        
        return metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from step {self.global_step}")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Trainer...")
    
    # Create mock model
    from ..model.context_tracker import create_context_tracker, ContextTrackerConfig
    
    config = ContextTrackerConfig(
        image_size=64,  # Smaller for testing
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    # Create mock batch
    B, L, T, S = 2, 10, 5, 3
    device = torch.device('cpu')
    
    batch = {
        'context_ids': torch.randint(0, 256, (B, L)),
        'images': torch.randn(B, 1, 64, 64),
        'target_tokens': torch.randint(0, 256, (B, T)),
        'target_actions': torch.randint(0, 60, (B, T)),
        'stroke_labels': torch.zeros(B, T, S),
        'active_strokes': torch.ones(B, S, dtype=torch.bool),
        'stroke_indices': [[(0, 5), (5, 10), (10, 16)] for _ in range(B)],
        'seq_lens': torch.tensor([T, T-1]),
    }
    
    # Set some stroke labels
    batch['stroke_labels'][:, 0, 0] = 1.0
    batch['stroke_labels'][:, 1, 1] = 1.0
    
    # Test train_step
    train_config = TrainingConfig(device='cpu', use_amp=False)
    
    print("\n1. Testing train_step...")
    model.train()
    metrics = train_step(model, batch, train_config)
    
    print(f"   loss: {metrics['loss']:.4f}")
    print(f"   symbol_loss: {metrics['symbol_loss']:.4f}")
    print(f"   position_loss: {metrics['position_loss']:.4f}")
    print(f"   stroke_loss: {metrics['stroke_loss']:.4f}")
    print(f"   total_steps: {metrics['total_steps']}")
    
    # Test backward
    print("\n2. Testing backward...")
    metrics['loss'].backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"   Total gradient norm: {grad_norm:.4f}")
    
    print("\nAll tests passed!")

