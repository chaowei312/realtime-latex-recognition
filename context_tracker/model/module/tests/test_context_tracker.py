"""
Comprehensive Tests for Context Tracker Model

Tests:
1. Model construction and configuration
2. Forward pass with all heads
3. Option B training step (incremental stroke removal)
4. Gradient flow through all components
5. Generation (AR decoding)
6. Edge cases and stress tests
"""

import torch
import torch.nn.functional as F
from typing import Dict, List

# Import model
import sys
sys.path.insert(0, str(__file__).split('context_tracker')[0])

from context_tracker.model.context_tracker import (
    ContextTrackerConfig,
    ContextTrackerModel,
    create_context_tracker,
    TextEmbedding,
    TransformerDecoder,
    SymbolHead,
)
from context_tracker.training.trainer import train_step, TrainingConfig


def test_model_construction():
    """Test model construction with various configs."""
    print("\n" + "=" * 60)
    print("TEST: Model Construction")
    print("=" * 60)
    
    # Default config
    config = ContextTrackerConfig()
    model = create_context_tracker(config)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Default config: {params:,} parameters")
    
    # Small config
    small_config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    small_model = create_context_tracker(small_config)
    small_params = sum(p.numel() for p in small_model.parameters())
    print(f"  Small config: {small_params:,} parameters")
    
    # Large config
    large_config = ContextTrackerConfig(
        image_size=128,
        d_model=512,
        num_heads=8,
        num_layers=6,
        vocab_size=1024
    )
    large_model = create_context_tracker(large_config)
    large_params = sum(p.numel() for p in large_model.parameters())
    print(f"  Large config: {large_params:,} parameters")
    
    # Check components exist
    assert hasattr(model, 'visual_encoder'), "Missing visual_encoder"
    assert hasattr(model, 'text_embedding'), "Missing text_embedding"
    assert hasattr(model, 'decoder'), "Missing decoder"
    assert hasattr(model, 'symbol_head'), "Missing symbol_head"
    assert hasattr(model, 'smm'), "Missing smm"
    assert hasattr(model, 'tpm'), "Missing tpm"
    
    print("  PASSED!")


def test_forward_pass():
    """Test forward pass with all heads."""
    print("\n" + "=" * 60)
    print("TEST: Forward Pass")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    B = 2
    device = torch.device('cpu')
    
    # Inputs
    images = torch.randn(B, 1, 64, 64)
    context_ids = torch.randint(0, 256, (B, 10))
    
    # Forward without stroke info
    output = model(images, context_ids)
    
    assert 'symbol_logits' in output, "Missing symbol_logits"
    assert 'action_probs' in output, "Missing action_probs"
    assert output['symbol_logits'].shape == (B, 256), f"Wrong symbol shape: {output['symbol_logits'].shape}"
    
    print(f"  symbol_logits: {output['symbol_logits'].shape}")
    print(f"  action_probs: {output['action_probs'].shape}")
    
    # Forward with stroke info
    stroke_indices = [(0, 5), (5, 10), (10, 16)]
    active_strokes = torch.ones(B, 3, dtype=torch.bool)
    
    output = model(
        images, context_ids,
        stroke_indices=[stroke_indices, stroke_indices],
        active_strokes=active_strokes,
        return_all=True
    )
    
    assert 'stroke_scores' in output, "Missing stroke_scores"
    assert output['stroke_scores'].shape == (B, 3), f"Wrong stroke shape: {output['stroke_scores'].shape}"
    
    print(f"  stroke_scores: {output['stroke_scores'].shape}")
    print(f"  h_t: {output['h_t'].shape}")
    print(f"  h_vc: {output['h_vc'].shape}")
    
    print("  PASSED!")


def test_loss_computation():
    """Test loss computation for all heads."""
    print("\n" + "=" * 60)
    print("TEST: Loss Computation")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    B = 2
    
    # Inputs
    images = torch.randn(B, 1, 64, 64)
    context_ids = torch.randint(0, 256, (B, 10))
    stroke_indices = [(0, 5), (5, 10), (10, 16)]
    active_strokes = torch.ones(B, 3, dtype=torch.bool)
    
    # Forward
    output = model(
        images, context_ids,
        stroke_indices=[stroke_indices, stroke_indices],
        active_strokes=active_strokes
    )
    
    # Targets
    target_tokens = torch.randint(0, 256, (B,))
    target_actions = torch.randint(0, 60, (B,))
    target_strokes = torch.zeros(B, 3)
    target_strokes[:, 0] = 1.0
    
    # Compute loss
    losses = model.compute_loss(
        output,
        target_tokens=target_tokens,
        target_actions=target_actions,
        target_stroke_labels=target_strokes,
        active_strokes=active_strokes
    )
    
    assert 'symbol_loss' in losses, "Missing symbol_loss"
    assert 'position_loss' in losses, "Missing position_loss"
    assert 'stroke_loss' in losses, "Missing stroke_loss"
    assert 'total_loss' in losses, "Missing total_loss"
    
    print(f"  symbol_loss: {losses['symbol_loss'].item():.4f}")
    print(f"  position_loss: {losses['position_loss'].item():.4f}")
    print(f"  stroke_loss: {losses['stroke_loss'].item():.4f}")
    print(f"  total_loss: {losses['total_loss'].item():.4f}")
    
    # Check losses are reasonable
    assert losses['total_loss'] > 0, "Total loss should be positive"
    assert not torch.isnan(losses['total_loss']), "Loss is NaN"
    assert not torch.isinf(losses['total_loss']), "Loss is Inf"
    
    print("  PASSED!")


def test_gradient_flow():
    """Test gradient flow through all components."""
    print("\n" + "=" * 60)
    print("TEST: Gradient Flow")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    B = 2
    
    # Inputs
    images = torch.randn(B, 1, 64, 64, requires_grad=True)
    context_ids = torch.randint(0, 256, (B, 10))
    stroke_indices = [(0, 5), (5, 10), (10, 16)]
    active_strokes = torch.ones(B, 3, dtype=torch.bool)
    
    # Forward
    output = model(
        images, context_ids,
        stroke_indices=[stroke_indices, stroke_indices],
        active_strokes=active_strokes
    )
    
    # Targets
    target_tokens = torch.randint(0, 256, (B,))
    target_actions = torch.randint(0, 60, (B,))
    target_strokes = torch.zeros(B, 3)
    target_strokes[:, 0] = 1.0
    
    # Compute loss
    losses = model.compute_loss(
        output,
        target_tokens=target_tokens,
        target_actions=target_actions,
        target_stroke_labels=target_strokes,
        active_strokes=active_strokes
    )
    
    # Backward
    losses['total_loss'].backward()
    
    # Check gradients
    modules_with_grad = set()
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            module_name = name.split('.')[0]
            modules_with_grad.add(module_name)
    
    print(f"  Modules with gradients: {sorted(modules_with_grad)}")
    
    expected_modules = {'visual_encoder', 'text_embedding', 'decoder', 'symbol_head', 'smm', 'tpm'}
    assert modules_with_grad >= expected_modules, f"Missing gradients for: {expected_modules - modules_with_grad}"
    
    # Check input gradient
    assert images.grad is not None, "Input should have gradient"
    print(f"  Input grad norm: {images.grad.norm().item():.4f}")
    
    print("  PASSED!")


def test_option_b_training():
    """Test Option B training step (incremental stroke removal)."""
    print("\n" + "=" * 60)
    print("TEST: Option B Training Step")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    B, L, T, S = 2, 10, 5, 3
    
    # Create batch
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
    
    # Set stroke labels (each token uses one stroke)
    for t in range(min(T, S)):
        batch['stroke_labels'][:, t, t] = 1.0
    
    # Train step
    train_config = TrainingConfig(device='cpu', use_amp=False)
    
    model.train()
    metrics = train_step(model, batch, train_config)
    
    print(f"  loss: {metrics['loss']:.4f}")
    print(f"  symbol_loss: {metrics['symbol_loss']:.4f}")
    print(f"  position_loss: {metrics['position_loss']:.4f}")
    print(f"  stroke_loss: {metrics['stroke_loss']:.4f}")
    print(f"  total_steps: {metrics['total_steps']}")
    
    # Check metrics
    assert metrics['loss'] > 0, "Loss should be positive"
    assert metrics['total_steps'] == sum(batch['seq_lens'].tolist()), "Wrong step count"
    
    # Test backward
    metrics['loss'].backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients should be non-zero"
    
    print("  PASSED!")


def test_generation():
    """Test autoregressive generation."""
    print("\n" + "=" * 60)
    print("TEST: Generation")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    model.eval()
    
    B = 2
    images = torch.randn(B, 1, 64, 64)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            images,
            max_length=10,
            temperature=1.0
        )
    
    assert 'tokens' in output, "Missing tokens"
    assert 'actions' in output, "Missing actions"
    assert output['tokens'].shape[0] == B, "Wrong batch size"
    assert output['tokens'].shape[1] <= 10, "Generated too many tokens"
    
    print(f"  Generated tokens shape: {output['tokens'].shape}")
    print(f"  Generated actions shape: {output['actions'].shape}")
    print(f"  First sample tokens: {output['tokens'][0].tolist()}")
    
    print("  PASSED!")


def test_attention_mask():
    """Test attention mask construction."""
    print("\n" + "=" * 60)
    print("TEST: Attention Mask")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    B = 1
    context_len = 5
    num_patches = 16  # Smaller for 64x64
    device = torch.device('cpu')
    
    mask = model._create_attention_mask(context_len, num_patches, B, device)
    
    total_len = context_len + num_patches + 2  # +2 for VC, INT
    assert mask.shape == (B, total_len, total_len), f"Wrong mask shape: {mask.shape}"
    
    # Check context is causal
    context_mask = mask[0, :context_len, :context_len]
    expected_causal = torch.tril(torch.ones(context_len, context_len))
    assert torch.allclose(context_mask, expected_causal), "Context should be causal"
    
    # Check VC only sees patches
    vc_pos = context_len + num_patches
    vc_mask = mask[0, vc_pos, :]
    assert vc_mask[:context_len].sum() == 0, "VC should not see context"
    assert vc_mask[context_len:vc_pos].sum() == num_patches, "VC should see all patches"
    
    # Check INT sees context + VC
    int_pos = vc_pos + 1
    int_mask = mask[0, int_pos, :]
    assert int_mask[:context_len].sum() == context_len, "INT should see all context"
    assert int_mask[vc_pos] == 1, "INT should see VC"
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Context causal: OK")
    print(f"  VC sees patches only: OK")
    print(f"  INT sees context + VC: OK")
    
    print("  PASSED!")


def test_edge_cases():
    """Test edge cases and stress tests."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    config = ContextTrackerConfig(
        image_size=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    # Empty context
    print("  Testing empty context...")
    images = torch.randn(1, 1, 64, 64)
    context_ids = torch.tensor([[config.bos_token_id]])  # Just BOS
    output = model(images, context_ids)
    assert output['symbol_logits'].shape == (1, 256), "Should handle empty context"
    print("    OK")
    
    # Long context
    print("  Testing long context...")
    long_context = torch.randint(0, 256, (1, 100))
    output = model(images, long_context)
    assert output['symbol_logits'].shape == (1, 256), "Should handle long context"
    print("    OK")
    
    # Single stroke
    print("  Testing single stroke...")
    stroke_indices = [(0, 16)]
    active_strokes = torch.ones(1, 1, dtype=torch.bool)
    output = model(
        images, context_ids,
        stroke_indices=[stroke_indices],
        active_strokes=active_strokes
    )
    assert output['stroke_scores'].shape == (1, 1), "Should handle single stroke"
    print("    OK")
    
    # Batch size 1
    print("  Testing batch size 1...")
    output = model(images, context_ids)
    assert output['symbol_logits'].shape[0] == 1, "Should handle batch size 1"
    print("    OK")
    
    # Large batch
    print("  Testing large batch...")
    large_images = torch.randn(16, 1, 64, 64)
    large_context = torch.randint(0, 256, (16, 10))
    output = model(large_images, large_context)
    assert output['symbol_logits'].shape[0] == 16, "Should handle large batch"
    print("    OK")
    
    print("  PASSED!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE CONTEXT TRACKER TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_model_construction,
        test_forward_pass,
        test_loss_computation,
        test_gradient_flow,
        test_option_b_training,
        test_generation,
        test_attention_mask,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

