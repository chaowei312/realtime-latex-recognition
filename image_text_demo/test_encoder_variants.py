"""
Test Script for Encoder Variants

Compares different Conv2D encoder configurations on the stroke letter dataset.
Uses JSON configs from context_tracker/configs/

Usage:
    python test_encoder_variants.py --variant stacked_3x3_s2 --epochs 10
    python test_encoder_variants.py --compare-all --epochs 5
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.stroke_letter_dataset.pytorch_dataset import (
    StrokeLetterDataset, 
    create_dataloaders,
    collate_images,
)

from context_tracker.model.module import (
    StrokePatchEncoder,
    load_encoder_from_config,
    list_available_variants,
)


class StrokeClassifier(nn.Module):
    """Simple classifier using configurable encoder."""
    
    def __init__(
        self,
        encoder_variant: str,
        num_classes: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = StrokePatchEncoder(
            variant=encoder_variant,
            embed_dim=embed_dim,
            input_channels=1,
            input_size=64,  # Stroke images are 64x64
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def train_variant(
    variant: str,
    data_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    embed_dim: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Train and evaluate a single variant."""
    
    print(f"\n{'='*60}")
    print(f"Training variant: {variant}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        mode='image',
        image_size=(64, 64),
        augment_train=True,
    )
    
    num_classes = train_loader.dataset.num_classes
    print(f"Classes: {num_classes}, Train samples: {len(train_loader.dataset)}")
    
    # Create model
    model = StrokeClassifier(
        encoder_variant=variant,
        num_classes=num_classes,
        embed_dim=embed_dim,
    ).to(device)
    
    # Print encoder config
    config = model.encoder.get_config_summary()
    print(f"Encoder: {config['name']}")
    print(f"  Type: {config['type']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Params: {config['num_params']:,}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model params: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    train_times = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        train_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f} | "
              f"Time={epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest: Loss={test_loss:.4f}, Acc={test_acc:.4f}")
    
    return {
        "variant": variant,
        "config": config,
        "total_params": total_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "avg_epoch_time": sum(train_times) / len(train_times),
    }


def compare_all_variants(
    data_dir: str,
    num_epochs: int = 5,
    batch_size: int = 32,
    embed_dim: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Compare all available encoder variants."""
    
    variants = list_available_variants()
    print(f"Comparing {len(variants)} encoder variants...")
    print(f"Device: {device}")
    print(f"Epochs per variant: {num_epochs}")
    
    results = []
    
    for variant in variants:
        try:
            result = train_variant(
                variant=variant,
                data_dir=data_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                embed_dim=embed_dim,
                device=device,
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR with {variant}: {e}")
            results.append({
                "variant": variant,
                "error": str(e),
            })
    
    # Summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Variant':<25} {'Params':<12} {'Val Acc':<10} {'Test Acc':<10} {'Time/Epoch':<10}")
    print("-"*80)
    
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<25} ERROR: {r['error']}")
        else:
            print(f"{r['variant']:<25} {r['total_params']:<12,} {r['best_val_acc']:<10.4f} "
                  f"{r['test_acc']:<10.4f} {r['avg_epoch_time']:<10.1f}s")
    
    print("="*80)
    
    # Find best
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['test_acc'])
        print(f"\nBest variant: {best['variant']} (Test Acc: {best['test_acc']:.4f})")
    
    # Save results
    output_file = Path(__file__).parent / "encoder_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test encoder variants on stroke letter dataset")
    parser.add_argument("--variant", type=str, default="stacked_3x3_s2",
                        help="Encoder variant to test")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all available variants")
    parser.add_argument("--list-variants", action="store_true",
                        help="List available variants and exit")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, cpu, or auto)")
    parser.add_argument("--data-dir", type=str, 
                        default=str(Path(__file__).parent / "data" / "stroke_letter_dataset"),
                        help="Path to stroke letter dataset")
    
    args = parser.parse_args()
    
    # List variants
    if args.list_variants:
        variants = list_available_variants()
        print("Available encoder variants:")
        for v in variants:
            encoder = StrokePatchEncoder(variant=v, embed_dim=256)
            config = encoder.get_config_summary()
            print(f"  - {v}: {config['name']} ({config['num_params']:,} params)")
        return
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Compare all or single
    if args.compare_all:
        compare_all_variants(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            embed_dim=args.embed_dim,
            device=device,
        )
    else:
        train_variant(
            variant=args.variant,
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embed_dim=args.embed_dim,
            device=device,
        )


if __name__ == "__main__":
    main()

