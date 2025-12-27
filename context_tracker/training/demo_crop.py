"""
Demo: Actual crop from MathWriting expression

Shows how training data is constructed:
1. Full expression: "2x+y/z" 
2. Context (text): "x+y/z" (already committed - no visual)
3. Edit region (image): Only "2" rendered with its position info

This demonstrates:
- Per-symbol bounding boxes from MathWriting
- How we crop the edit symbol
- Positional information (2 is LEFT of x)
"""

import numpy as np
from pathlib import Path

# Try to import visualization
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available - text output only")

from context_tracker.data.mathwriting_atomic import (
    MathWritingChunk,
    Stroke,
    StrokeBBox,
    ChunkRenderer,
)


def create_mock_expression_2x_plus_y_over_z():
    """
    Create a mock MathWriting expression: 2x + y/z
    
    This mimics what real MathWriting data looks like:
    - Each symbol has strokes with coordinates
    - Each symbol has a bounding box
    - We can extract individual symbols for editing
    """
    
    # Symbol strokes (mock handwriting coordinates)
    # Each stroke is a sequence of (x, y) points
    
    # "2" - at position x=0-20
    stroke_2 = Stroke(points=np.array([
        [5, 10], [15, 5], [20, 15], [10, 25], [5, 25], [20, 30]
    ], dtype=np.float32))
    
    # "x" - at position x=25-45
    stroke_x_1 = Stroke(points=np.array([
        [25, 10], [45, 30]
    ], dtype=np.float32))
    stroke_x_2 = Stroke(points=np.array([
        [25, 30], [45, 10]
    ], dtype=np.float32))
    
    # "+" - at position x=50-70
    stroke_plus_h = Stroke(points=np.array([
        [50, 20], [70, 20]
    ], dtype=np.float32))
    stroke_plus_v = Stroke(points=np.array([
        [60, 10], [60, 30]
    ], dtype=np.float32))
    
    # "y" - at position x=75-95, y=0-20 (numerator)
    stroke_y = Stroke(points=np.array([
        [75, 5], [85, 15], [85, 15], [95, 5],
        [85, 15], [80, 25]
    ], dtype=np.float32))
    
    # "/" - fraction bar at position x=75-95, y=25-30
    stroke_frac = Stroke(points=np.array([
        [75, 27], [95, 27]
    ], dtype=np.float32))
    
    # "z" - at position x=75-95, y=30-50 (denominator)
    stroke_z = Stroke(points=np.array([
        [75, 35], [95, 35], [75, 50], [95, 50]
    ], dtype=np.float32))
    
    # All strokes
    all_strokes = [
        stroke_2,           # 0
        stroke_x_1,         # 1
        stroke_x_2,         # 2
        stroke_plus_h,      # 3
        stroke_plus_v,      # 4
        stroke_y,           # 5
        stroke_frac,        # 6
        stroke_z            # 7
    ]
    
    # Per-symbol bounding boxes (KEY for cropping!)
    symbol_bboxes = [
        StrokeBBox(symbol="2", stroke_indices=[0], 
                   x_min=5, y_min=5, x_max=20, y_max=30),
        StrokeBBox(symbol="x", stroke_indices=[1, 2], 
                   x_min=25, y_min=10, x_max=45, y_max=30),
        StrokeBBox(symbol="+", stroke_indices=[3, 4], 
                   x_min=50, y_min=10, x_max=70, y_max=30),
        StrokeBBox(symbol="y", stroke_indices=[5], 
                   x_min=75, y_min=5, x_max=95, y_max=25),
        StrokeBBox(symbol="/", stroke_indices=[6], 
                   x_min=75, y_min=25, x_max=95, y_max=30),  # fraction bar
        StrokeBBox(symbol="z", stroke_indices=[7], 
                   x_min=75, y_min=30, x_max=95, y_max=50),
    ]
    
    # Full chunk
    full_chunk = MathWritingChunk(
        latex="2x+\\frac{y}{z}",
        strokes=all_strokes,
        symbol_bboxes=symbol_bboxes,
        sample_id="demo_2x_plus_y_over_z"
    )
    
    return full_chunk


def demo_edit_scenario():
    """
    Demonstrate the edit scenario:
    
    Context: "x + y/z" (already in text, symbols 1-5)
    Edit: Add "2" to the left of x (symbol 0)
    
    Model sees:
    - Context as TEXT tokens: ["x", "+", "y", "/", "z"]
    - Edit image: Only "2" rendered
    - Position info: "2" bbox is LEFT of "x" bbox
    """
    
    print("=" * 70)
    print("DEMO: Edit Scenario from MathWriting Expression")
    print("=" * 70)
    
    # Create full expression
    full_chunk = create_mock_expression_2x_plus_y_over_z()
    
    print(f"\n1. FULL EXPRESSION")
    print(f"   LaTeX: {full_chunk.latex}")
    print(f"   Total strokes: {full_chunk.num_strokes}")
    print(f"   Symbols: {[b.symbol for b in full_chunk.symbol_bboxes]}")
    
    # Show bounding boxes
    print(f"\n2. PER-SYMBOL BOUNDING BOXES (from MathWriting)")
    print("   " + "-" * 60)
    print(f"   {'Symbol':<10} {'Strokes':<15} {'BBox (x_min, y_min, x_max, y_max)'}")
    print("   " + "-" * 60)
    for bbox in full_chunk.symbol_bboxes:
        print(f"   {bbox.symbol:<10} {str(bbox.stroke_indices):<15} "
              f"({bbox.x_min:.0f}, {bbox.y_min:.0f}, {bbox.x_max:.0f}, {bbox.y_max:.0f})")
    
    # Scenario: Context = "x + y/z", Edit = "2"
    print(f"\n3. EDIT SCENARIO")
    print(f"   Context (TEXT, already committed): x + y / z")
    print(f"   Edit region (IMAGE): Symbol '2'")
    print(f"   Expected output: '2' at position LEFT of 'x'")
    
    # Extract edit symbol
    edit_symbol_idx = 0  # "2"
    edit_chunk = full_chunk.extract_atomic(edit_symbol_idx)
    
    print(f"\n4. EXTRACTED EDIT CHUNK")
    print(f"   Symbol: '{edit_chunk.latex}'")
    print(f"   Strokes: {edit_chunk.num_strokes}")
    print(f"   BBox: {edit_chunk.bbox}")
    
    # Position relative to next symbol
    edit_bbox = full_chunk.symbol_bboxes[0]  # "2"
    next_bbox = full_chunk.symbol_bboxes[1]  # "x"
    
    print(f"\n5. POSITIONAL RELATIONSHIP")
    print(f"   '2' bbox x_range: [{edit_bbox.x_min:.0f}, {edit_bbox.x_max:.0f}]")
    print(f"   'x' bbox x_range: [{next_bbox.x_min:.0f}, {next_bbox.x_max:.0f}]")
    print(f"   Relationship: '2' is LEFT of 'x' (x_max={edit_bbox.x_max:.0f} < x_min={next_bbox.x_min:.0f})")
    
    # For TPM: action would be (x, LEFT) or in our system: (x_idx, RIGHT reversed)
    # But in our tree model: we're ADDING "2", so action = (x, LEFT_OF) 
    # Or in sequence: context=[x,+,y,/,z], we insert 2 at position 0
    
    print(f"\n6. TRAINING LABELS")
    print(f"   target_token: '2'")
    print(f"   target_action: Insert LEFT of 'x' in tree")
    print(f"   stroke_labels: [1] (select the '2' stroke group)")
    
    # Render images
    if HAS_PIL:
        print(f"\n7. RENDERED IMAGES")
        
        renderer = ChunkRenderer(image_size=128, line_width=3)
        
        # Render full expression
        full_image = renderer.render(full_chunk, return_numpy=True)
        
        # Render edit region only
        edit_image = renderer.render(edit_chunk, return_numpy=True)
        
        print(f"   Full image shape: {full_image.shape}")
        print(f"   Edit image shape: {edit_image.shape}")
        
        # Save images for inspection
        output_dir = Path("context_tracker/training/demo_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(full_image).save(output_dir / "full_expression.png")
        Image.fromarray(edit_image).save(output_dir / "edit_region.png")
        
        print(f"\n   Saved to: {output_dir}/")
        print(f"   - full_expression.png (2x+y/z)")
        print(f"   - edit_region.png (just '2')")
    
    # Show what model receives during training
    print(f"\n8. WHAT MODEL RECEIVES")
    print("   " + "=" * 50)
    print(f"   CONTEXT (text tokens, KV-cached):")
    print(f"     tokens = ['<BOS>', 'x', '+', 'y', '/', 'z']")
    print(f"     (No visual patches - these are committed!)")
    print()
    print(f"   EDIT REGION (image patches):")
    print(f"     image = render('2')  # 128x128 grayscale")
    print(f"     patch_embeddings = VisualEncoder(image)  # [64, 256]")
    print(f"     VC_token = pool(patch_embeddings)  # [256]")
    print()
    print(f"   STROKE GROUPS:")
    print(f"     Group 0: '2' strokes -> patches [0-20]")
    print(f"     (No artifacts in this simple example)")
    print()
    print(f"   EXPECTED OUTPUT:")
    print(f"     Head A (Symbol): '2'")
    print(f"     Head B (SMM): stroke_scores = [1.0] (select group 0)")
    print(f"     Head C (TPM): action = (insert LEFT of x)")
    print("   " + "=" * 50)
    
    return full_chunk, edit_chunk


def demo_with_artifact():
    """
    Demo with artifact stroke (incomplete symbol).
    
    Same expression but model also sees an incomplete "/" stroke
    that should be IGNORED by SMM.
    """
    
    print("\n" + "=" * 70)
    print("DEMO: Edit with Artifact Stroke")
    print("=" * 70)
    
    full_chunk = create_mock_expression_2x_plus_y_over_z()
    
    # Extract "2" for edit
    edit_chunk = full_chunk.extract_atomic(0)
    
    # Create artifact: incomplete "/" (just the start of the stroke)
    artifact_stroke = Stroke(points=np.array([
        [30, 35], [35, 30]  # Just 2 points - incomplete
    ], dtype=np.float32))
    
    print(f"\n1. EDIT CHUNK: '{edit_chunk.latex}'")
    print(f"   Strokes: {edit_chunk.num_strokes}")
    
    print(f"\n2. ARTIFACT: Incomplete '/' stroke")
    print(f"   Points: {artifact_stroke.num_points} (incomplete)")
    
    print(f"\n3. WHAT MODEL SEES IN IMAGE:")
    print(f"   - Complete '2' strokes (TARGET)")
    print(f"   - Incomplete '/' strokes (ARTIFACT - should ignore)")
    
    print(f"\n4. STROKE GROUPS:")
    print(f"   Group 0: '2' complete    -> is_target=1")
    print(f"   Group 1: '/' incomplete  -> is_target=0 (artifact)")
    
    print(f"\n5. SMM EXPECTED SCORES:")
    print(f"   When outputting '2':")
    print(f"     Group 0 score: ~1.0 (select)")
    print(f"     Group 1 score: ~0.0 (ignore - it's incomplete!)")
    
    # Render with artifact
    if HAS_PIL:
        renderer = ChunkRenderer(image_size=128, line_width=3)
        
        # Render edit + artifact
        combined_image = renderer.render(edit_chunk, artifact_strokes=[artifact_stroke], return_numpy=True)
        
        output_dir = Path("context_tracker/training/demo_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(combined_image).save(output_dir / "edit_with_artifact.png")
        
        print(f"\n   Saved: edit_with_artifact.png")
        print(f"   (Shows '2' + incomplete '/' artifact)")


def demo_fraction_edit():
    """
    Demo editing within a fraction structure.
    
    Expression: 2x + y/z
    Context: 2x + /z (y is being edited/added)
    Edit: Add "y" as numerator
    """
    
    print("\n" + "=" * 70)
    print("DEMO: Fraction Edit (adding numerator)")
    print("=" * 70)
    
    full_chunk = create_mock_expression_2x_plus_y_over_z()
    
    # "y" is symbol index 3
    edit_chunk = full_chunk.extract_atomic(3)
    
    y_bbox = full_chunk.symbol_bboxes[3]
    frac_bbox = full_chunk.symbol_bboxes[4]  # "/"
    z_bbox = full_chunk.symbol_bboxes[5]
    
    print(f"\n1. EDITING: Add 'y' as numerator in fraction")
    print(f"   Full expression: {full_chunk.latex}")
    print(f"   Context (text): 2x + _ / z  (y is missing)")
    print(f"   Edit: Insert 'y'")
    
    print(f"\n2. POSITIONAL ANALYSIS:")
    print(f"   'y' bbox: y_range=[{y_bbox.y_min:.0f}, {y_bbox.y_max:.0f}]")
    print(f"   '/' bbox: y_range=[{frac_bbox.y_min:.0f}, {frac_bbox.y_max:.0f}]")
    print(f"   'z' bbox: y_range=[{z_bbox.y_min:.0f}, {z_bbox.y_max:.0f}]")
    print()
    print(f"   'y' is ABOVE '/' (y_max={y_bbox.y_max:.0f} < frac_y_min={frac_bbox.y_min:.0f})")
    print(f"   'z' is BELOW '/' (z_y_min={z_bbox.y_min:.0f} > frac_y_max={frac_bbox.y_max:.0f})")
    
    print(f"\n3. TPM ACTION:")
    print(f"   action = (frac, ABOVE)  # Insert y above fraction bar")
    print(f"   This tells the tree: y is numerator of the fraction")
    
    print(f"\n4. RELATION TYPES:")
    print(f"   RIGHT: horizontal sequence (2 -> x -> +)")
    print(f"   ABOVE: numerator in fraction (y -> frac)")
    print(f"   BELOW: denominator in fraction (z -> frac)")
    print(f"   SUP: superscript (x^2)")
    print(f"   SUB: subscript (a_i)")


if __name__ == "__main__":
    # Run demos
    demo_edit_scenario()
    demo_with_artifact()
    demo_fraction_edit()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

