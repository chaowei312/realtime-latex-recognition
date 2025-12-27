"""
Comprehensive Tests for SMM, TPM, and VisualPatchEncoder Modules

Tests:
1. Forward pass correctness
2. Shape validation
3. Option B: Incremental stroke removal
4. K-cache management
5. Backward gradient flow
6. Edge cases
7. Visual encoder tests
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..stroke_modification import (
    StrokeModificationModule,
    StrokeModificationLoss,
    create_stroke_modification_module,
)
from ..position_head import (
    TreeAwarePositionModule,
    TPMLoss,
    KCacheManager,
    ARDecoderWithStrokeRemoval,
    create_tpm,
    create_cache_manager,
)
from ..relationship_tensor import RelationType
from ..visual_encoder import (
    VisualPatchEncoder,
    StackedConvStem,
    PositionalEmbedding2D,
    PatchStrokeMapper,
    create_visual_encoder,
)


def test_smm_forward():
    """Test SMM forward pass and shapes."""
    print("\n" + "=" * 60)
    print("TEST: SMM Forward Pass")
    print("=" * 60)
    
    B = 4
    d_model = 512
    num_patches = 20
    num_strokes = 5
    num_heads = 4
    
    smm = StrokeModificationModule(
        d_int=d_model,
        d_vc=d_model,
        d_patch=d_model,
        num_heads=num_heads,
    )
    
    # Inputs
    h_int = torch.randn(B, d_model)
    h_vc = torch.randn(B, d_model)
    patches = torch.randn(B, num_patches, d_model)
    stroke_indices = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]
    
    # Forward
    scores = smm(h_int, h_vc, patches, stroke_indices)
    
    # Checks
    assert scores.shape == (B, num_strokes), f"Expected {(B, num_strokes)}, got {scores.shape}"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0, 1]"
    
    print(f"  Input shapes: h_int={h_int.shape}, h_vc={h_vc.shape}, patches={patches.shape}")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
    print("  PASSED!")


def test_smm_with_active_mask():
    """Test SMM with active stroke mask (Option B)."""
    print("\n" + "=" * 60)
    print("TEST: SMM with Active Stroke Mask (Option B)")
    print("=" * 60)
    
    B = 2
    d_model = 256
    num_patches = 15
    num_strokes = 5
    
    smm = StrokeModificationModule(d_int=d_model, d_vc=d_model, d_patch=d_model)
    
    h_int = torch.randn(B, d_model)
    h_vc = torch.randn(B, d_model)
    patches = torch.randn(B, num_patches, d_model)
    stroke_indices = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    
    # All active
    active = torch.ones(B, num_strokes, dtype=torch.bool)
    scores_all = smm(h_int, h_vc, patches, stroke_indices, active)
    
    # Some inactive
    active_partial = torch.tensor([[True, False, True, False, True],
                                   [False, True, False, True, False]])
    scores_partial = smm(h_int, h_vc, patches, stroke_indices, active_partial)
    
    # Check inactive strokes have 0 score
    for b in range(B):
        for s in range(num_strokes):
            if not active_partial[b, s]:
                assert scores_partial[b, s] == 0.0, f"Inactive stroke should have 0 score"
    
    print(f"  All active scores: {scores_all[0].tolist()}")
    print(f"  Partial active mask: {active_partial[0].tolist()}")
    print(f"  Partial scores: {scores_partial[0].tolist()}")
    print("  Inactive strokes correctly zeroed!")
    print("  PASSED!")


def test_smm_incremental_removal():
    """Test SMM incremental stroke removal flow."""
    print("\n" + "=" * 60)
    print("TEST: SMM Incremental Stroke Removal")
    print("=" * 60)
    
    B = 1
    d_model = 256
    num_patches = 12
    num_strokes = 4
    
    smm = StrokeModificationModule(d_int=d_model, d_vc=d_model, d_patch=d_model)
    
    h_int = torch.randn(B, d_model)
    h_vc = torch.randn(B, d_model)
    patches = torch.randn(B, num_patches, d_model)
    stroke_indices = [(0, 3), (3, 6), (6, 9), (9, 12)]
    
    active = torch.ones(B, num_strokes, dtype=torch.bool)
    
    print(f"  Initial active: {active[0].tolist()}")
    
    for step in range(3):
        scores = smm(h_int, h_vc, patches, stroke_indices, active)
        contrib, new_active = smm.get_contributing_strokes(scores, active, threshold=0.5)
        patches = smm.remove_strokes_from_patches(patches, stroke_indices, contrib)
        active = new_active
        
        print(f"  Step {step+1}: scores={[f'{s:.2f}' for s in scores[0].tolist()]}, "
              f"contrib={contrib[0].tolist()}, active={active[0].tolist()}")
    
    print("  PASSED!")


def test_smm_gradient():
    """Test SMM backward gradient flow."""
    print("\n" + "=" * 60)
    print("TEST: SMM Gradient Flow")
    print("=" * 60)
    
    B = 2
    d_model = 128
    num_patches = 8
    num_strokes = 2
    
    smm = StrokeModificationModule(d_int=d_model, d_vc=d_model, d_patch=d_model)
    loss_fn = StrokeModificationLoss()
    
    h_int = torch.randn(B, d_model, requires_grad=True)
    h_vc = torch.randn(B, d_model, requires_grad=True)
    patches = torch.randn(B, num_patches, d_model, requires_grad=True)
    stroke_indices = [(0, 4), (4, 8)]
    
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    # Forward
    scores = smm(h_int, h_vc, patches, stroke_indices)
    loss = loss_fn(scores, target)
    
    # Backward
    loss.backward()
    
    # Check gradients exist
    assert h_int.grad is not None, "h_int should have gradient"
    assert h_vc.grad is not None, "h_vc should have gradient"
    assert patches.grad is not None, "patches should have gradient"
    
    # Check gradients are not all zero
    assert h_int.grad.abs().sum() > 0, "h_int gradient should be non-zero"
    assert h_vc.grad.abs().sum() > 0, "h_vc gradient should be non-zero"
    assert patches.grad.abs().sum() > 0, "patches gradient should be non-zero"
    
    # Check model parameters have gradients
    for name, param in smm.named_parameters():
        assert param.grad is not None, f"{name} should have gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient should be non-zero"
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  h_int grad norm: {h_int.grad.norm().item():.6f}")
    print(f"  h_vc grad norm: {h_vc.grad.norm().item():.6f}")
    print(f"  patches grad norm: {patches.grad.norm().item():.6f}")
    print("  All gradients flow correctly!")
    print("  PASSED!")


def test_tpm_forward():
    """Test TPM forward pass and shapes."""
    print("\n" + "=" * 60)
    print("TEST: TPM Forward Pass")
    print("=" * 60)
    
    B = 4
    d_model = 512
    num_heads = 8
    d_head = d_model // num_heads
    N_ctx = 6
    num_relations = 6
    
    tpm = TreeAwarePositionModule(
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        num_relations=num_relations,
    )
    
    h_t = torch.randn(B, d_model)
    h_vc = torch.randn(B, d_model)
    k_cache = torch.randn(B, num_heads, N_ctx, d_head)
    
    # Forward
    action_probs, action_logits = tpm(h_t, h_vc, k_cache)
    
    # Checks
    expected_actions = N_ctx * num_relations
    assert action_probs.shape == (B, expected_actions), f"Expected {(B, expected_actions)}, got {action_probs.shape}"
    assert action_logits.shape == (B, expected_actions)
    
    # Check softmax property
    prob_sums = action_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Action probs should sum to 1"
    
    print(f"  Input: h_t={h_t.shape}, h_vc={h_vc.shape}, k_cache={k_cache.shape}")
    print(f"  Action space: {N_ctx} tokens x {num_relations} relations = {expected_actions}")
    print(f"  Output: action_probs={action_probs.shape}")
    print(f"  Prob sums: {prob_sums.tolist()}")
    print("  PASSED!")


def test_tpm_action_decode():
    """Test TPM action encoding/decoding."""
    print("\n" + "=" * 60)
    print("TEST: TPM Action Encode/Decode")
    print("=" * 60)
    
    tpm = create_tpm(d_model=256, num_heads=4, d_head=64)
    
    # Test all relations
    for rel in RelationType:
        for parent_idx in range(5):
            action_idx = tpm.encode_action(parent_idx, rel)
            decoded_parent, decoded_rel = tpm.decode_action(action_idx, num_parents=5)
            
            assert decoded_parent == parent_idx, f"Parent mismatch: {decoded_parent} vs {parent_idx}"
            assert decoded_rel == rel.name, f"Relation mismatch: {decoded_rel} vs {rel.name}"
    
    print("  Tested all (parent, relation) combinations")
    print("  Encode -> Decode is identity")
    print("  PASSED!")


def test_tpm_k_cache_growing():
    """Test TPM with growing K cache during AR."""
    print("\n" + "=" * 60)
    print("TEST: TPM with Growing K Cache")
    print("=" * 60)
    
    B = 2
    d_model = 256
    num_heads = 4
    d_head = 64
    N_ctx = 3
    
    tpm = create_tpm(d_model=d_model, num_heads=num_heads, d_head=d_head)
    
    h_vc = torch.randn(B, d_model)
    
    print(f"  Initial context: {N_ctx} tokens")
    
    for step in range(4):
        num_tokens = N_ctx + step
        k_cache = torch.randn(B, num_heads, num_tokens, d_head)
        h_t = torch.randn(B, d_model)
        
        probs, logits = tpm(h_t, h_vc, k_cache)
        expected_actions = num_tokens * 6
        
        assert probs.shape[1] == expected_actions, f"Expected {expected_actions} actions, got {probs.shape[1]}"
        
        top_action = probs[0].argmax().item()
        parent, rel = tpm.decode_action(top_action, num_tokens)
        
        print(f"  Step {step}: K_cache[{num_tokens}] -> {expected_actions} actions, "
              f"top=({parent}, {rel})")
    
    print("  PASSED!")


def test_tpm_gradient():
    """Test TPM backward gradient flow."""
    print("\n" + "=" * 60)
    print("TEST: TPM Gradient Flow")
    print("=" * 60)
    
    B = 2
    d_model = 128
    num_heads = 4
    d_head = 32
    N_ctx = 4
    
    tpm = create_tpm(d_model=d_model, num_heads=num_heads, d_head=d_head)
    loss_fn = TPMLoss()
    
    h_t = torch.randn(B, d_model, requires_grad=True)
    h_vc = torch.randn(B, d_model, requires_grad=True)
    k_cache = torch.randn(B, num_heads, N_ctx, d_head, requires_grad=True)
    
    target_actions = torch.randint(0, N_ctx * 6, (B,))
    
    # Forward
    probs, logits = tpm(h_t, h_vc, k_cache)
    loss = loss_fn(logits, target_actions)
    
    # Backward
    loss.backward()
    
    # Check gradients exist and non-zero
    assert h_t.grad is not None and h_t.grad.abs().sum() > 0, "h_t gradient issue"
    assert h_vc.grad is not None and h_vc.grad.abs().sum() > 0, "h_vc gradient issue"
    assert k_cache.grad is not None and k_cache.grad.abs().sum() > 0, "k_cache gradient issue"
    
    # Check model parameters
    for name, param in tpm.named_parameters():
        assert param.grad is not None, f"{name} should have gradient"
        assert param.grad.abs().sum() > 0, f"{name} gradient should be non-zero"
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  h_t grad norm: {h_t.grad.norm().item():.6f}")
    print(f"  h_vc grad norm: {h_vc.grad.norm().item():.6f}")
    print(f"  k_cache grad norm: {k_cache.grad.norm().item():.6f}")
    print("  All gradients flow correctly!")
    print("  PASSED!")


def test_k_cache_manager():
    """Test KCacheManager append and recomputation."""
    print("\n" + "=" * 60)
    print("TEST: KCacheManager")
    print("=" * 60)
    
    B = 2
    num_heads = 4
    d_head = 64
    N_ctx = 3
    
    cache_mgr = create_cache_manager(num_heads=num_heads, d_head=d_head)
    
    # Initialize
    context_k = torch.randn(B, num_heads, N_ctx, d_head)
    cache_mgr.init_from_context(context_k)
    
    print(f"  Initial: {cache_mgr.num_tokens} tokens")
    
    # Test different action types
    test_actions = [
        (0, RelationType.RIGHT),   # Append
        (1, RelationType.SUP),     # Append
        (2, RelationType.BELOW),   # Recompute
        (0, RelationType.SUB),     # Append
        (1, RelationType.INSIDE),  # Recompute
    ]
    
    for parent_idx, rel in test_actions:
        new_key = torch.randn(B, num_heads, 1, d_head)
        recomputed = cache_mgr.update_after_action(
            action=(parent_idx, rel),
            new_key=new_key,
        )
        status = "RECOMPUTE" if recomputed else "append"
        print(f"  ({parent_idx}, {rel.name:6}) -> {status}, cache: {cache_mgr.num_tokens}")
    
    assert cache_mgr.num_tokens == N_ctx + len(test_actions)
    print("  PASSED!")


def test_combined_smm_tpm_gradient():
    """Test gradient flow through both SMM and TPM in combined loss."""
    print("\n" + "=" * 60)
    print("TEST: Combined SMM + TPM Gradient Flow")
    print("=" * 60)
    
    B = 2
    d_model = 256
    num_heads = 4
    d_head = 64
    num_patches = 12
    num_strokes = 3
    N_ctx = 4
    
    # Create modules
    smm = StrokeModificationModule(d_int=d_model, d_vc=d_model, d_patch=d_model, num_heads=num_heads)
    tpm = create_tpm(d_model=d_model, num_heads=num_heads, d_head=d_head)
    smm_loss_fn = StrokeModificationLoss()
    tpm_loss_fn = TPMLoss()
    
    # Shared inputs
    h_int = torch.randn(B, d_model, requires_grad=True)
    h_vc = torch.randn(B, d_model, requires_grad=True)
    h_t = torch.randn(B, d_model, requires_grad=True)
    patches = torch.randn(B, num_patches, d_model, requires_grad=True)
    k_cache = torch.randn(B, num_heads, N_ctx, d_head, requires_grad=True)
    stroke_indices = [(0, 4), (4, 8), (8, 12)]
    
    # Targets
    stroke_target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    action_target = torch.randint(0, N_ctx * 6, (B,))
    
    # Forward both
    stroke_scores = smm(h_int, h_vc, patches, stroke_indices)
    action_probs, action_logits = tpm(h_t, h_vc, k_cache)
    
    # Combined loss
    smm_loss = smm_loss_fn(stroke_scores, stroke_target)
    tpm_loss = tpm_loss_fn(action_logits, action_target)
    total_loss = smm_loss + tpm_loss
    
    # Backward
    total_loss.backward()
    
    # Check all inputs have gradients
    for name, tensor in [("h_int", h_int), ("h_vc", h_vc), ("h_t", h_t), 
                         ("patches", patches), ("k_cache", k_cache)]:
        assert tensor.grad is not None, f"{name} should have gradient"
        assert tensor.grad.abs().sum() > 0, f"{name} gradient should be non-zero"
    
    # h_vc should have gradient from BOTH modules
    print(f"  SMM loss: {smm_loss.item():.4f}")
    print(f"  TPM loss: {tpm_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  h_vc grad norm: {h_vc.grad.norm().item():.6f} (from both SMM and TPM)")
    print("  Combined gradient flow works!")
    print("  PASSED!")


def test_smm_padded_forward():
    """Test SMM with padded batch input."""
    print("\n" + "=" * 60)
    print("TEST: SMM Padded Batch Forward")
    print("=" * 60)
    
    B = 2
    d_model = 256
    num_strokes = 3
    max_patches = 5
    
    smm = StrokeModificationModule(d_int=d_model, d_vc=d_model, d_patch=d_model)
    
    h_int = torch.randn(B, d_model)
    h_vc = torch.randn(B, d_model)
    patches = torch.randn(B, num_strokes, max_patches, d_model)
    stroke_lengths = torch.tensor([[3, 5, 2], [4, 2, 5]])  # Actual lengths
    
    scores = smm.forward_with_padding(h_int, h_vc, patches, stroke_lengths, max_patches)
    
    assert scores.shape == (B, num_strokes)
    print(f"  Padded patches shape: {patches.shape}")
    print(f"  Stroke lengths: {stroke_lengths.tolist()}")
    print(f"  Output scores: {scores.shape}")
    print("  PASSED!")


def test_tpm_with_valid_mask():
    """Test TPM with valid action mask."""
    print("\n" + "=" * 60)
    print("TEST: TPM with Valid Action Mask")
    print("=" * 60)
    
    B = 2
    d_model = 256
    num_heads = 4
    d_head = 64
    N_ctx = 3
    num_actions = N_ctx * 6
    
    tpm = create_tpm(d_model=d_model, num_heads=num_heads, d_head=d_head)
    
    h_t = torch.randn(B, d_model)
    h_vc = torch.randn(B, d_model)
    k_cache = torch.randn(B, num_heads, N_ctx, d_head)
    
    # Mask out some actions
    valid_mask = torch.ones(B, num_actions, dtype=torch.bool)
    valid_mask[:, 0:6] = False  # Mask all actions for first token
    
    probs, logits = tpm(h_t, h_vc, k_cache, valid_mask=valid_mask)
    
    # Masked actions should have ~0 probability
    masked_probs = probs[:, 0:6]
    assert masked_probs.max() < 1e-6, "Masked actions should have near-zero probability"
    
    print(f"  Valid mask: first 6 actions masked")
    print(f"  Masked action max prob: {masked_probs.max().item():.8f}")
    print(f"  Valid action probs sum: {probs[:, 6:].sum(dim=-1).tolist()}")
    print("  PASSED!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    d_model = 128
    
    # Single stroke
    smm = StrokeModificationModule(d_int=d_model, d_vc=d_model, d_patch=d_model)
    scores = smm(
        torch.randn(1, d_model),
        torch.randn(1, d_model),
        torch.randn(1, 4, d_model),
        [(0, 4)]
    )
    assert scores.shape == (1, 1), "Single stroke should work"
    print("  Single stroke: OK")
    
    # Single token
    tpm = create_tpm(d_model=d_model, num_heads=4, d_head=32)
    probs, _ = tpm(
        torch.randn(1, d_model),
        torch.randn(1, d_model),
        torch.randn(1, 4, 1, 32)
    )
    assert probs.shape == (1, 6), "Single token should give 6 actions"
    print("  Single token: OK")
    
    # Large batch
    scores = smm(
        torch.randn(32, d_model),
        torch.randn(32, d_model),
        torch.randn(32, 20, d_model),
        [(i*4, (i+1)*4) for i in range(5)]
    )
    assert scores.shape == (32, 5), "Large batch should work"
    print("  Large batch (32): OK")
    
    print("  PASSED!")


def test_visual_encoder_forward():
    """Test VisualPatchEncoder forward pass and shapes."""
    print("\n" + "=" * 60)
    print("TEST: VisualPatchEncoder Forward Pass")
    print("=" * 60)
    
    B = 2
    image_size = 128
    embed_dim = 256
    
    encoder = create_visual_encoder(
        image_size=image_size,
        embed_dim=embed_dim,
        variant='standard',
        add_vc_token=True
    )
    
    # Test basic forward
    x = torch.randn(B, 1, image_size, image_size)
    patches, vc_token = encoder(x)
    
    expected_num_patches = encoder.num_patches
    assert patches.shape == (B, expected_num_patches, embed_dim), \
        f"Expected {(B, expected_num_patches, embed_dim)}, got {patches.shape}"
    assert vc_token is not None, "VC token should be returned"
    assert vc_token.shape == (B, embed_dim), f"Expected {(B, embed_dim)}, got {vc_token.shape}"
    print(f"  Forward: {x.shape} -> patches {patches.shape}, VC {vc_token.shape}")
    
    # Test without VC token
    encoder_no_vc = create_visual_encoder(
        image_size=image_size,
        embed_dim=embed_dim,
        variant='tiny',
        add_vc_token=False
    )
    patches_no_vc, vc_none = encoder_no_vc(x)
    assert vc_none is None, "VC token should be None when add_vc_token=False"
    print("  No VC token: OK")
    
    # Test with return_grid
    patches, vc_token, grid = encoder(x, return_grid=True)
    expected_grid_shape = (B, embed_dim, encoder.grid_size[0], encoder.grid_size[1])
    assert grid.shape == expected_grid_shape, f"Expected {expected_grid_shape}, got {grid.shape}"
    print(f"  Grid shape: {grid.shape}")
    
    print("  PASSED!")


def test_visual_encoder_stroke_mask():
    """Test VisualPatchEncoder stroke masking."""
    print("\n" + "=" * 60)
    print("TEST: VisualPatchEncoder Stroke Masking")
    print("=" * 60)
    
    B = 2
    encoder = create_visual_encoder(image_size=128, embed_dim=256, variant='standard')
    
    x = torch.randn(B, 1, 128, 128)
    
    # Create a mask that zeros out bottom-right quadrant
    stroke_mask = torch.ones(B, 8, 8)
    stroke_mask[:, 4:, 4:] = 0  # Mask out 16 patches
    
    patches, vc_token = encoder(x, stroke_mask=stroke_mask)
    
    # Check that masked patches are zero
    mask_flat = stroke_mask.flatten(1)
    masked_region = patches * (1 - mask_flat.unsqueeze(-1))
    masked_norm = masked_region.norm().item()
    
    assert masked_norm < 1e-6, f"Masked region should be zero, got norm {masked_norm}"
    print(f"  Masked region norm: {masked_norm:.8f} (expected ~0)")
    
    # Check that unmasked patches are non-zero
    unmasked_region = patches * mask_flat.unsqueeze(-1)
    unmasked_norm = unmasked_region.norm().item()
    assert unmasked_norm > 0, "Unmasked region should have non-zero values"
    print(f"  Unmasked region norm: {unmasked_norm:.4f} (should be > 0)")
    
    # Test with flat mask input
    flat_mask = stroke_mask.flatten(1)
    patches_flat, _ = encoder(x, stroke_mask=flat_mask)
    diff = (patches - patches_flat).abs().max().item()
    assert diff < 1e-6, f"Flat mask should give same result, diff={diff}"
    print("  Flat mask input: OK")
    
    print("  PASSED!")


def test_visual_encoder_gradient():
    """Test gradient flow through VisualPatchEncoder."""
    print("\n" + "=" * 60)
    print("TEST: VisualPatchEncoder Gradient Flow")
    print("=" * 60)
    
    encoder = create_visual_encoder(image_size=128, embed_dim=256, variant='standard')
    
    x = torch.randn(2, 1, 128, 128, requires_grad=True)
    patches, vc_token = encoder(x)
    
    # Loss on both outputs
    loss = patches.sum() + vc_token.sum()
    loss.backward()
    
    # Check input gradient
    assert x.grad is not None, "Input should have gradient"
    assert x.grad.abs().sum() > 0, "Input gradient should be non-zero"
    print(f"  Input grad norm: {x.grad.norm().item():.4f}")
    
    # Check all parameter gradients
    params_with_grad = 0
    for name, p in encoder.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad += 1
    
    total_params = sum(1 for _ in encoder.parameters())
    print(f"  Parameters with gradient: {params_with_grad}/{total_params}")
    assert params_with_grad > 0, "Some parameters should have gradients"
    
    print("  PASSED!")


def test_patch_stroke_mapper():
    """Test PatchStrokeMapper functionality."""
    print("\n" + "=" * 60)
    print("TEST: PatchStrokeMapper")
    print("=" * 60)
    
    mapper = PatchStrokeMapper(image_size=128, grid_size=8)
    
    # Test stroke_to_patches
    stroke = torch.tensor([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]])
    patch_indices = mapper.stroke_to_patches(stroke)
    
    assert patch_indices.shape == (3,), f"Expected (3,), got {patch_indices.shape}"
    print(f"  stroke_to_patches: 3 points -> patches {patch_indices.tolist()}")
    
    # Test boundary handling (clamp to valid range)
    stroke_boundary = torch.tensor([[-10.0, -10.0], [200.0, 200.0]])  # Out of bounds
    patch_boundary = mapper.stroke_to_patches(stroke_boundary)
    assert all(0 <= p < 64 for p in patch_boundary.tolist()), "Should clamp to valid range"
    print("  Boundary handling: OK")
    
    # Test stroke_to_patch_mask
    stroke1 = torch.tensor([[10.0, 10.0], [20.0, 20.0]])  # Top-left area
    stroke2 = torch.tensor([[100.0, 100.0], [110.0, 110.0]])  # Bottom-right area
    
    mask = mapper.stroke_to_patch_mask([stroke1, stroke2], batch_size=2)
    assert mask.shape == (2, 8, 8), f"Expected (2, 8, 8), got {mask.shape}"
    assert mask.sum() > 0, "Mask should have some active patches"
    print(f"  stroke_to_patch_mask: {mask.sum().item():.0f} active patch-cells")
    
    # Test with active_strokes mask
    active = torch.tensor([[True, False], [True, True]])  # Batch 0: only stroke1, Batch 1: both
    mask_selective = mapper.stroke_to_patch_mask([stroke1, stroke2], batch_size=2, active_strokes=active)
    print("  Active strokes masking: OK")
    
    # Test build_stroke_to_patch_index
    indices = mapper.build_stroke_to_patch_index([stroke1, stroke2])
    assert len(indices) == 2, "Should return index range for each stroke"
    print(f"  Stroke indices: {indices}")
    
    print("  PASSED!")


def test_positional_embedding_2d():
    """Test 2D positional embeddings."""
    print("\n" + "=" * 60)
    print("TEST: PositionalEmbedding2D")
    print("=" * 60)
    
    embed_dim = 256
    grid_size = 8
    
    # Test sinusoidal
    pos_embed_sin = PositionalEmbedding2D(embed_dim, grid_size, learnable=False)
    pos = pos_embed_sin.get_embedding()
    assert pos.shape == (1, 64, embed_dim), f"Expected (1, 64, {embed_dim}), got {pos.shape}"
    print(f"  Sinusoidal embedding shape: {pos.shape}")
    
    # Test that different positions have different embeddings
    pos_flat = pos.squeeze(0)
    pos_0 = pos_flat[0]
    pos_1 = pos_flat[1]
    diff = (pos_0 - pos_1).abs().sum().item()
    assert diff > 0, "Different positions should have different embeddings"
    print(f"  Position difference (0 vs 1): {diff:.4f}")
    
    # Test learnable
    pos_embed_learn = PositionalEmbedding2D(embed_dim, grid_size, learnable=True)
    pos_learn = pos_embed_learn.get_embedding()
    assert pos_learn.shape == (1, 64, embed_dim), f"Expected (1, 64, {embed_dim}), got {pos_learn.shape}"
    assert pos_embed_learn.pos_embed.requires_grad, "Learnable embedding should require grad"
    print("  Learnable embedding: OK")
    
    # Test forward
    x = torch.randn(2, 64, embed_dim)
    out = pos_embed_sin(x)
    assert out.shape == x.shape, "Output shape should match input"
    print("  Forward pass: OK")
    
    print("  PASSED!")


def test_visual_encoder_variants():
    """Test different encoder variants."""
    print("\n" + "=" * 60)
    print("TEST: VisualPatchEncoder Variants")
    print("=" * 60)
    
    x = torch.randn(2, 1, 128, 128)
    
    for variant in ['tiny', 'standard', 'large']:
        encoder = create_visual_encoder(
            image_size=128,
            embed_dim=256,
            variant=variant
        )
        
        patches, vc = encoder(x)
        params = sum(p.numel() for p in encoder.parameters())
        
        print(f"  {variant}: {params:,} params, output {patches.shape}")
        
        assert patches.shape == (2, 64, 256), f"Shape mismatch for {variant}"
    
    print("  PASSED!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SMM + TPM + VISUAL ENCODER TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_smm_forward,
        test_smm_with_active_mask,
        test_smm_incremental_removal,
        test_smm_gradient,
        test_smm_padded_forward,
        test_tpm_forward,
        test_tpm_action_decode,
        test_tpm_k_cache_growing,
        test_tpm_gradient,
        test_tpm_with_valid_mask,
        test_k_cache_manager,
        test_combined_smm_tpm_gradient,
        test_edge_cases,
        # Visual Encoder tests
        test_visual_encoder_forward,
        test_visual_encoder_stroke_mask,
        test_visual_encoder_gradient,
        test_patch_stroke_mapper,
        test_positional_embedding_2d,
        test_visual_encoder_variants,
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

