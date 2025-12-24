#!/usr/bin/env python
"""
Demo script for stroke-level rendering and dataset generation.

Uses captured stroke data from stroke_corpus/annotations/.

Run: python stroke_demo.py
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Core data structures
from stroke_primitives import (
    Stroke, StrokeSequence,
    RollingSnowball,
    PatchToken, ModelInput, ModelOutput,
    StrokeAttentionAnalyzer, infer_consumed_strokes, process_model_output,
)

# Rendering and dataset generation
from scripts.stroke_renderer import (
    StrokeDataLoader,
    StrokeRenderer,
    AugmentationConfig,
    get_symbol_info,
    get_available_symbols,
)
from scripts.stroke_dataset import (
    StrokeDatasetGenerator,
    StrokeExample,
)


def demo_stroke_data():
    """Demo: Load and explore captured stroke data."""
    print("=" * 70)
    print("DEMO 1: Captured Stroke Data")
    print("=" * 70)
    
    loader = StrokeDataLoader()
    stats = loader.get_statistics()
    
    print(f"\nDataset loaded:")
    print(f"  Symbols: {stats['num_symbols']}")
    print(f"  Total strokes: {stats['total_strokes']}")
    print(f"  Total variations: {stats['total_variations']}")
    print(f"  Avg strokes/symbol: {stats['avg_strokes_per_symbol']:.1f}")
    print(f"  Avg variations/stroke: {stats['avg_variations_per_stroke']:.1f}")
    
    # Show some symbols
    print("\n  Sample symbols:")
    symbols = get_available_symbols()[:10]
    for sym in symbols:
        info = get_symbol_info(sym)
        if info:
            print(f"    '{sym}' ({info['latex']}): {info['num_strokes']} strokes, "
                  f"{info['total_variations']} combinations")


def demo_rendering():
    """Demo: Render symbols using captured stroke data."""
    print("\n" + "=" * 70)
    print("DEMO 2: Symbol Rendering")
    print("=" * 70)
    
    renderer = StrokeRenderer(canvas_size=128)
    
    test_symbols = ['A', 'x', '+', '0']
    print("\n  Rendering symbols with augmentation:")
    
    for sym in test_symbols:
        info = get_symbol_info(sym)
        if info:
            canvas, points = renderer.render_symbol(sym, augment=True, return_points=True)
            print(f"    '{sym}': canvas={canvas.shape}, points={len(points)}, "
                  f"variations={info['total_variations']}")
        else:
            print(f"    '{sym}': Not in dataset")
    
    # Show augmentation config
    print("\n  Augmentation settings:")
    aug = AugmentationConfig()
    print(f"    Jitter: {aug.jitter_amount}")
    print(f"    Rotation: {aug.rotation_range}")
    print(f"    Scale: {aug.scale_range}")
    print(f"    Thickness: {aug.thickness_range}")


def demo_stroke_progression():
    """Demo: Generate stroke progression (WAIT vs complete)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Stroke Progression (WAIT Logic)")
    print("=" * 70)
    
    gen = StrokeDatasetGenerator(seed=42, canvas_size=128)
    
    # Find a multi-stroke symbol
    for sym in get_available_symbols():
        info = get_symbol_info(sym)
        if info and info['num_strokes'] >= 2:
            print(f"\n  Symbol '{sym}' ({info['num_strokes']} strokes):")
            
            examples = gen.generate_stroke_progression(sym, context="f(x) = ")
            for ex in examples:
                status = "COMPLETE" if ex.is_complete else "WAIT"
                print(f"    Stroke {ex.stroke_idx + 1}/{ex.total_strokes}: "
                      f"output='{ex.output}' [{status}]")
            break


def demo_rolling_snowball():
    """Demo: Rolling snowball patch accumulation."""
    print("\n" + "=" * 70)
    print("DEMO 4: Rolling Snowball Patch Accumulation")
    print("=" * 70)
    
    snowball = RollingSnowball()
    
    print("\n  Simulating: Draw multi-stroke symbol with noise")
    
    # First stroke
    stroke1 = Stroke(0, 0.2, 0.3, 0.35, 0.45, 0.0, 0.2)
    result = snowball.add_stroke(stroke1)
    print(f"  1. First stroke: buffer={result['buffer_size']}, groups={result['num_groups']}")
    
    # Noise dot (different location)
    noise = Stroke(1, 0.7, 0.5, 0.72, 0.52, 0.3, 0.05)
    result = snowball.add_stroke(noise)
    print(f"  2. Noise dot: buffer={result['buffer_size']}, groups={result['num_groups']}")
    
    # Second stroke (near first)
    stroke2 = Stroke(2, 0.22, 0.32, 0.38, 0.48, 0.5, 0.2)
    result = snowball.add_stroke(stroke2)
    print(f"  3. Second stroke: buffer={result['buffer_size']}, groups={result['num_groups']}")
    
    # Consume the symbol
    print("\n  >>> Model recognizes symbol - consuming strokes 0, 2...")
    consume = snowball.consume_strokes("x", group_index=0)
    print(f"  Consumed: {consume['consumed_count']}, Remaining: {consume['remaining_count']}")
    print(f"  Noise stroke still in buffer: {consume['remaining_count'] > 0}")


def demo_model_io():
    """Demo: Model input/output structure."""
    print("\n" + "=" * 70)
    print("DEMO 5: Model Input/Output Structure")
    print("=" * 70)
    
    # Create input
    model_input = ModelInput(latex_context="x + ", latex_token_ids=[43, 120])
    model_input.add_stroke_patches(0, [
        PatchToken(stroke_id=0, patch_idx=0, bbox=(0.2, 0.3, 0.3, 0.4)),
        PatchToken(stroke_id=0, patch_idx=1, bbox=(0.25, 0.35, 0.35, 0.45)),
    ])
    model_input.add_stroke_patches(1, [
        PatchToken(stroke_id=1, patch_idx=0, bbox=(0.7, 0.5, 0.72, 0.52)),
    ])
    
    print(f"\n  Input state:")
    print(f"    Context: '{model_input.latex_context}'")
    print(f"    Strokes: {model_input.total_strokes}")
    print(f"    Patches: {model_input.total_patches}")
    
    # Simulate recognition
    output = ModelOutput.recognize(r"\alpha", stroke_ids=[0], confidence=0.95)
    new_input = process_model_output(model_input, output)
    
    print(f"\n  After recognition:")
    print(f"    Output: '{output.latex_output}'")
    print(f"    Consumed strokes: {output.consumed_stroke_ids}")
    print(f"    New context: '{new_input.latex_context}'")
    print(f"    Remaining strokes: {new_input.total_strokes}")


def demo_balanced_dataset():
    """Demo: Generate balanced training dataset."""
    print("\n" + "=" * 70)
    print("DEMO 6: Balanced Dataset Generation")
    print("=" * 70)
    
    gen = StrokeDatasetGenerator(seed=42)
    
    examples = gen.generate_balanced(total_count=50, wait_ratio=0.4)
    
    wait_count = sum(1 for e in examples if e.output == "<WAIT>")
    complete_count = len(examples) - wait_count
    
    print(f"\n  Generated {len(examples)} examples:")
    print(f"    <WAIT>: {wait_count} ({wait_count/len(examples)*100:.0f}%)")
    print(f"    Complete: {complete_count} ({complete_count/len(examples)*100:.0f}%)")
    
    # Show sample
    print("\n  Sample examples:")
    for ex in examples[:5]:
        print(f"    {ex.symbol}: {ex.stroke_idx+1}/{ex.total_strokes} -> '{ex.output}'")


def main():
    """Run all demos."""
    demo_stroke_data()
    demo_rendering()
    demo_stroke_progression()
    demo_rolling_snowball()
    demo_model_io()
    demo_balanced_dataset()
    
    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)
    print("\nKey modules:")
    print("  - stroke_level.py: Data structures (Stroke, RollingSnowball, ModelInput/Output)")
    print("  - scripts/stroke_renderer.py: GPU rendering with captured data")
    print("  - scripts/stroke_dataset.py: Training dataset generation")


if __name__ == "__main__":
    main()
