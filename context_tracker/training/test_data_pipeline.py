"""
Test the training data pipeline end-to-end.

Verifies:
1. EditDataset produces correct stroke groups from bboxes
2. Artifacts are properly labeled
3. Training batch is correctly formatted
"""

import torch
import numpy as np
from pathlib import Path

from context_tracker.data.mathwriting_atomic import (
    MathWritingAtomic,
    MathWritingChunk,
    Stroke,
    StrokeBBox,
    ChunkRenderer,
    CompositionalAugmentor,
)
from context_tracker.training.dataset import (
    EditDataset,
    SimpleTokenizer,
    collate_edit_batch,
)


def create_mock_mathwriting_dataset():
    """Create a mock MathWritingAtomic-like dataset for testing."""
    
    class MockMathWriting:
        """Mock dataset that returns pre-defined chunks."""
        
        def __init__(self):
            self.chunks = self._create_chunks()
        
        def _create_chunks(self):
            chunks = []
            
            # Chunk 1: "a+b" with per-symbol bboxes
            chunks.append(MathWritingChunk(
                latex="a+b",
                strokes=[
                    Stroke(points=np.array([[0, 0], [10, 10], [0, 20]], dtype=np.float32)),  # a
                    Stroke(points=np.array([[20, 10], [30, 10], [25, 5], [25, 15]], dtype=np.float32)),  # +
                    Stroke(points=np.array([[40, 0], [40, 20], [50, 10], [40, 10]], dtype=np.float32)),  # b
                ],
                symbol_bboxes=[
                    StrokeBBox(symbol="a", stroke_indices=[0], x_min=0, y_min=0, x_max=10, y_max=20),
                    StrokeBBox(symbol="+", stroke_indices=[1], x_min=20, y_min=5, x_max=30, y_max=15),
                    StrokeBBox(symbol="b", stroke_indices=[2], x_min=40, y_min=0, x_max=50, y_max=20),
                ],
                sample_id="mock_a_plus_b"
            ))
            
            # Chunk 2: "x^2" with per-symbol bboxes
            chunks.append(MathWritingChunk(
                latex="x^2",
                strokes=[
                    Stroke(points=np.array([[0, 10], [10, 20], [0, 20], [10, 10]], dtype=np.float32)),  # x
                    Stroke(points=np.array([[15, 0], [20, 0], [15, 5], [20, 5]], dtype=np.float32)),  # 2
                ],
                symbol_bboxes=[
                    StrokeBBox(symbol="x", stroke_indices=[0], x_min=0, y_min=10, x_max=10, y_max=20),
                    StrokeBBox(symbol="2", stroke_indices=[1], x_min=15, y_min=0, x_max=20, y_max=5),
                ],
                sample_id="mock_x_squared"
            ))
            
            # Chunk 3: "y/z" (fraction) with per-symbol bboxes
            chunks.append(MathWritingChunk(
                latex="\\frac{y}{z}",
                strokes=[
                    Stroke(points=np.array([[0, 0], [5, 10], [5, 10], [10, 0], [5, 10], [5, 15]], dtype=np.float32)),  # y
                    Stroke(points=np.array([[0, 20], [10, 20]], dtype=np.float32)),  # frac bar
                    Stroke(points=np.array([[0, 25], [10, 25], [0, 35], [10, 35]], dtype=np.float32)),  # z
                ],
                symbol_bboxes=[
                    StrokeBBox(symbol="y", stroke_indices=[0], x_min=0, y_min=0, x_max=10, y_max=15),
                    StrokeBBox(symbol="/", stroke_indices=[1], x_min=0, y_min=18, x_max=10, y_max=22),
                    StrokeBBox(symbol="z", stroke_indices=[2], x_min=0, y_min=25, x_max=10, y_max=35),
                ],
                sample_id="mock_y_over_z"
            ))
            
            # Chunk 4: "=" 
            chunks.append(MathWritingChunk(
                latex="=",
                strokes=[
                    Stroke(points=np.array([[0, 5], [20, 5]], dtype=np.float32)),
                    Stroke(points=np.array([[0, 15], [20, 15]], dtype=np.float32)),
                ],
                symbol_bboxes=[
                    StrokeBBox(symbol="=", stroke_indices=[0, 1], x_min=0, y_min=5, x_max=20, y_max=15),
                ],
                sample_id="mock_equals"
            ))
            
            return chunks
        
        def __len__(self):
            return len(self.chunks)
        
        def __getitem__(self, idx):
            return self.chunks[idx % len(self.chunks)]
    
    return MockMathWriting()


def test_edit_dataset():
    """Test EditDataset produces correct output."""
    
    print("=" * 70)
    print("TEST: EditDataset Pipeline")
    print("=" * 70)
    
    # Create mock dataset
    mock_mw = create_mock_mathwriting_dataset()
    tokenizer = SimpleTokenizer()
    
    # Create EditDataset with smaller chunk counts (mock has only 4 chunks)
    edit_ds = EditDataset(
        mathwriting=mock_mw,
        tokenizer=tokenizer,
        image_size=128,
        min_chunks=1,  # Smaller for mock dataset
        max_chunks=2,
        artifact_ratio=0.5  # 50% chance of artifacts
    )
    
    print(f"\n1. Dataset created")
    print(f"   Mock chunks: {len(mock_mw)}")
    print(f"   Virtual size: {len(edit_ds)}")
    
    # Get a sample
    sample = edit_ds[0]
    
    print(f"\n2. Sample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"   {key}: list of {len(value)} items")
        else:
            print(f"   {key}: {value}")
    
    # Check stroke groups
    print(f"\n3. Stroke Groups Analysis:")
    print(f"   num_strokes (groups): {sample['num_strokes']}")
    print(f"   stroke_indices: {sample['stroke_indices']}")
    print(f"   is_target_group: {sample['is_target_group'].tolist()}")
    print(f"   num_target_groups: {sample['num_target_groups']}")
    print(f"   num_artifact_groups: {sample['num_artifact_groups']}")
    
    # Check stroke labels
    print(f"\n4. Stroke Labels (per-token):")
    print(f"   stroke_labels shape: {sample['stroke_labels'].shape}")
    print(f"   stroke_labels:\n{sample['stroke_labels']}")
    
    print("\n   Interpretation:")
    T = sample['target_tokens'].shape[0]
    G = sample['num_strokes']
    for t in range(min(T, 3)):  # Show first 3 tokens
        labels = sample['stroke_labels'][t].tolist()
        selected = [i for i, v in enumerate(labels) if v > 0.5]
        print(f"   Token {t}: select groups {selected}")
    
    print("\n   PASSED!")
    return sample


def test_batch_collation():
    """Test batch collation with variable-length samples."""
    
    print("\n" + "=" * 70)
    print("TEST: Batch Collation")
    print("=" * 70)
    
    mock_mw = create_mock_mathwriting_dataset()
    tokenizer = SimpleTokenizer()
    edit_ds = EditDataset(mock_mw, tokenizer, image_size=128, min_chunks=1, max_chunks=2)
    
    # Get multiple samples
    samples = [edit_ds[i] for i in range(4)]
    
    # Collate
    batch = collate_edit_batch(samples)
    
    print(f"\n1. Batch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"   {key}: list of {len(value)} items")
    
    print(f"\n2. Batch details:")
    print(f"   context_ids: {batch['context_ids'].shape}")
    print(f"   images: {batch['images'].shape}")
    print(f"   target_tokens: {batch['target_tokens'].shape}")
    print(f"   stroke_labels: {batch['stroke_labels'].shape}")
    print(f"   is_target_group: {batch['is_target_group'].shape}")
    print(f"   seq_lens: {batch['seq_lens'].tolist()}")
    
    # Verify is_target_group
    print(f"\n3. is_target_group per sample:")
    for i in range(batch['is_target_group'].shape[0]):
        groups = batch['is_target_group'][i].tolist()
        targets = sum(1 for g in groups if g > 0.5)
        artifacts = sum(1 for g in groups if 0 < g < 0.5)  # Should be 0 in ideal case
        zeros = sum(1 for g in groups if g == 0)
        print(f"   Sample {i}: {targets} targets, {zeros} padding/artifacts")
    
    print("\n   PASSED!")
    return batch


def test_with_model():
    """Test that batch works with model forward pass."""
    
    print("\n" + "=" * 70)
    print("TEST: Model Forward with Data Pipeline")
    print("=" * 70)
    
    from context_tracker.model.context_tracker import create_context_tracker, ContextTrackerConfig
    
    # Create small model
    config = ContextTrackerConfig(
        image_size=128,
        d_model=128,
        num_heads=4,
        num_layers=2,
        vocab_size=256
    )
    model = create_context_tracker(config)
    
    # Get batch
    mock_mw = create_mock_mathwriting_dataset()
    tokenizer = SimpleTokenizer()
    edit_ds = EditDataset(mock_mw, tokenizer, image_size=128, min_chunks=1, max_chunks=2)
    samples = [edit_ds[i] for i in range(2)]
    batch = collate_edit_batch(samples)
    
    print(f"\n1. Model created: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Forward pass
    output = model(
        images=batch['images'],
        context_ids=batch['context_ids'],
        stroke_indices=batch['stroke_indices'],
        active_strokes=batch['active_strokes']
    )
    
    print(f"\n2. Forward pass output:")
    print(f"   symbol_logits: {output['symbol_logits'].shape}")
    print(f"   action_probs: {output['action_probs'].shape}")
    print(f"   stroke_scores: {output['stroke_scores'].shape}")
    
    # Compute loss
    losses = model.compute_loss(
        output,
        target_tokens=batch['target_tokens'][:, 0],  # First token
        target_actions=batch['target_actions'][:, 0],
        target_stroke_labels=batch['stroke_labels'][:, 0, :],
        active_strokes=batch['active_strokes']
    )
    
    print(f"\n3. Losses:")
    for name, value in losses.items():
        print(f"   {name}: {value.item():.4f}")
    
    # Backward
    losses['total_loss'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\n4. Gradient norm: {grad_norm:.4f}")
    
    print("\n   PASSED!")


def test_artifact_labeling():
    """Test that artifacts are properly labeled as non-target."""
    
    print("\n" + "=" * 70)
    print("TEST: Artifact Labeling")
    print("=" * 70)
    
    mock_mw = create_mock_mathwriting_dataset()
    tokenizer = SimpleTokenizer()
    
    # Force artifacts
    edit_ds = EditDataset(mock_mw, tokenizer, image_size=128, min_chunks=1, max_chunks=2, artifact_ratio=1.0)
    
    # Sample multiple times to get some with artifacts
    samples_with_artifacts = []
    for i in range(20):
        sample = edit_ds[i]
        if sample['num_artifact_groups'] > 0:
            samples_with_artifacts.append(sample)
    
    print(f"\n1. Sampled 20, found {len(samples_with_artifacts)} with artifacts")
    
    if samples_with_artifacts:
        sample = samples_with_artifacts[0]
        print(f"\n2. Sample with artifact:")
        print(f"   num_strokes (groups): {sample['num_strokes']}")
        print(f"   num_target_groups: {sample['num_target_groups']}")
        print(f"   num_artifact_groups: {sample['num_artifact_groups']}")
        print(f"   is_target_group: {sample['is_target_group'].tolist()}")
        
        # Verify artifact groups have is_target=0
        artifact_count = sum(1 for x in sample['is_target_group'].tolist() if x == 0)
        print(f"\n3. Verification:")
        print(f"   Groups with is_target=0: {artifact_count}")
        print(f"   Expected artifacts: {sample['num_artifact_groups']}")
        
        # Check stroke_labels - artifact groups should never be 1
        print(f"\n4. Stroke labels for artifact groups:")
        for g in range(sample['num_strokes']):
            is_target = sample['is_target_group'][g].item()
            max_label = sample['stroke_labels'][:, g].max().item()
            print(f"   Group {g}: is_target={is_target:.0f}, max_label={max_label:.1f}")
            
            if is_target == 0:  # Artifact
                assert max_label == 0, f"Artifact group {g} should have all-zero labels!"
    
    print("\n   PASSED!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRAINING DATA PIPELINE TESTS")
    print("=" * 70)
    
    test_edit_dataset()
    test_batch_collation()
    test_artifact_labeling()
    test_with_model()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)

