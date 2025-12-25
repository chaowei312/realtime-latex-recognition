#!/usr/bin/env python
"""
Full Pipeline Demo: Generating Training Data for LaTeX Editing

This demo shows:
1. Generate atomic samples at specific depths (diagrams + math expressions)
2. Compositional augmentation to form context
3. Render atomic LaTeX to images
4. Edit symbols within samples
5. Use handwriting strokes (complete vs incomplete cases)

Run from the context_tracker/data directory:
    python -m demos.full_pipeline_demo
"""

import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_case_generator import (
    CaseGenerator, 
    EditCase, 
    ExpressionBuilder,
    analyze_latex_complexity,
    compute_case_depth,
)

# =============================================================================
# 1. GENERATE ATOMIC SAMPLES AT SPECIFIC DEPTH
# =============================================================================

def demo_atomic_generation():
    """Generate atomic samples at specific depths."""
    print("\n" + "=" * 70)
    print("1. GENERATING ATOMIC SAMPLES AT SPECIFIC DEPTHS")
    print("=" * 70)
    
    gen = CaseGenerator(seed=42)
    
    # Generate at exact depth
    print("\n[A] Math expressions at depth 2:")
    depth2_cases = gen.generate_at_depth(target_depth=2, count=5)
    for case in depth2_cases:
        print(f"  [{case.depth}] {case.operation}: {case.after_latex[:50]}...")
    
    print("\n[B] Math expressions at depth 4:")
    depth4_cases = gen.generate_at_depth(target_depth=4, count=5)
    for case in depth4_cases:
        print(f"  [{case.depth}] {case.operation}: {case.after_latex[:60]}...")
    
    # Generate diagrams using DiagramCaseGenerator
    print("\n[C] Commutative Diagrams:")
    from synthetic_case_generator import DiagramCaseGenerator
    diagram_cases = DiagramCaseGenerator.generate_add_cases(count=3)
    for case in diagram_cases:
        # Show first 80 chars of the tikzcd
        latex_preview = case.after_latex.replace('\n', ' ')[:80]
        print(f"  [{case.subcategory}] {latex_preview}...")
    
    # Generate curriculum with depth distribution
    print("\n[D] Curriculum with custom depth distribution:")
    curriculum_cases = gen.generate_curriculum(
        total_count=100,
        depth_distribution={
            1: 0.30,  # 30% simple
            2: 0.25,  # 25% 
            3: 0.20,  # 20%
            4: 0.15,  # 15%
            5: 0.10,  # 10% complex
        }
    )
    
    # Count depth distribution
    depth_counts = {}
    for case in curriculum_cases:
        depth_counts[case.depth] = depth_counts.get(case.depth, 0) + 1
    print(f"  Generated depth distribution: {dict(sorted(depth_counts.items()))}")
    
    return depth2_cases + depth4_cases + diagram_cases


# =============================================================================
# 2. COMPOSITIONAL AUGMENTATION (Forming Context)
# =============================================================================

def demo_compositional_augmentation(atomic_cases: List[EditCase]):
    """Demonstrate compositional augmentation to form context."""
    print("\n" + "=" * 70)
    print("2. COMPOSITIONAL AUGMENTATION (Forming Context)")
    print("=" * 70)
    
    try:
        from augmentation import ChunkPool, ContextComposer, ExpressionChunk
        from augmentation.chunk_types import EditInfo
    except ImportError:
        print("  [!] Augmentation module not fully available, showing concept...")
        # Demonstrate the concept manually
        print("\n  Concept: Combine atomic chunks into composed context\n")
        
        # Manual composition example
        chunks = [case.after_latex for case in atomic_cases[:3]]
        separator = " \\quad "
        composed = separator.join(chunks)
        
        print(f"  Chunk 1: {chunks[0][:40]}...")
        print(f"  Chunk 2: {chunks[1][:40]}...")
        print(f"  Chunk 3: {chunks[2][:40]}...")
        print(f"\n  Composed context:")
        print(f"    {composed[:80]}...")
        
        return composed
    
    # Full augmentation flow
    print("\n[A] Building chunk pool from atomic cases...")
    gen = CaseGenerator(seed=42)
    
    # Build pool from generator
    pool = ChunkPool.from_case_generator(gen, chunks_per_depth=20, max_depth=5)
    print(f"  Created pool with {len(pool.chunks)} chunks")
    print(f"  Depth distribution: {pool.get_stats()['by_depth']}")
    
    print("\n[B] Composing training examples...")
    composer = ContextComposer(pool, seed=42)
    
    # Compose with different context sizes
    examples = composer.compose_batch(
        num_examples=10,
        num_context_chunks=3,  # 3 chunks of context
        target_depth=2,
    )
    
    print(f"\n  Generated {len(examples)} composed examples:")
    for ex in examples[:3]:
        training_pair = ex.get_training_pair()
        print(f"\n  Example {ex.id}:")
        print(f"    Context: \"{training_pair['input_context'][:60]}...\"")
        print(f"    Edit: '{ex.edit_old_symbol}' -> '{ex.edit_new_symbol}'")
        print(f"    Output: \"{training_pair['output']}\"")
        print(f"    Edit position: {ex.edit_start_pos}:{ex.edit_end_pos}")
    
    return examples


# =============================================================================
# 3. RENDER ATOMIC LATEX TO IMAGES
# =============================================================================

def demo_latex_rendering(cases: List[EditCase]):
    """Render LaTeX expressions to images."""
    print("\n" + "=" * 70)
    print("3. RENDERING ATOMIC LATEX TO IMAGES")
    print("=" * 70)
    
    try:
        from scripts.latex_renderer import (
            render_latex, 
            render_latex_batch,
            get_render_capabilities,
            DiagramTemplates,
        )
    except ImportError as e:
        print(f"  [!] LaTeX renderer not available: {e}")
        return None
    
    # Check capabilities
    caps = get_render_capabilities()
    print(f"\n[A] Render capabilities:")
    print(f"  pdflatex: {'✓' if caps['pdflatex_available'] else '✗'}")
    print(f"  PyMuPDF: {'✓' if caps['pymupdf_available'] else '✗'}")
    print(f"  Can render: {'✓' if caps['can_render'] else '✗'}")
    
    if not caps['can_render']:
        print("\n  [!] Cannot render - install MiKTeX + pymupdf")
        return None
    
    # Render math expressions
    print("\n[B] Rendering math expressions:")
    expressions = [
        r"x^2 + y^2 = r^2",
        r"\frac{\partial f}{\partial x}",
        r"\int_0^\infty e^{-x} dx = 1",
        r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
    ]
    
    rendered_images = []
    for expr in expressions:
        img = render_latex(expr, dpi=150)
        if img is not None:
            rendered_images.append((expr, img))
            print(f"  ✓ {expr[:35]:35} -> shape {img.shape}")
        else:
            print(f"  ✗ {expr[:35]:35} -> FAILED")
    
    # Render diagram
    print("\n[C] Rendering commutative diagram:")
    diagram = DiagramTemplates.square()
    img = render_latex(diagram, dpi=150)
    if img is not None:
        print(f"  ✓ Commutative square -> shape {img.shape}")
        rendered_images.append((diagram, img))
    
    # Render from generated cases
    print("\n[D] Rendering from generated cases:")
    for case in cases[:3]:
        if 'tikzcd' not in case.after_latex:  # Skip diagrams for now
            img = render_latex(case.after_latex, dpi=150)
            if img is not None:
                print(f"  ✓ depth={case.depth} -> shape {img.shape}")
            else:
                print(f"  ✗ depth={case.depth} -> FAILED")
    
    return rendered_images


# =============================================================================
# 4. EDIT SYMBOLS WITHIN SAMPLES
# =============================================================================

def demo_symbol_editing():
    """Demonstrate editing symbols within expressions."""
    print("\n" + "=" * 70)
    print("4. EDITING SYMBOLS WITHIN SAMPLES")
    print("=" * 70)
    
    gen = CaseGenerator(seed=42)
    
    # Generate various edit types
    print("\n[A] REPLACE operations:")
    replace_cases = gen.generate_by_operation("REPLACE", count=5)
    for case in replace_cases[:3]:
        print(f"  Before: {case.before_latex[:50]}")
        print(f"  After:  {case.after_latex[:50]}")
        print(f"  Desc:   {case.edit_description}")
        print()
    
    print("\n[B] ADD operations:")
    add_cases = gen.generate_by_operation("ADD", count=5)
    for case in add_cases[:3]:
        print(f"  Before: {case.before_latex[:50]}")
        print(f"  After:  {case.after_latex[:50]}")
        print(f"  Desc:   {case.edit_description}")
        print()
    
    print("\n[C] FILL operations (incomplete structures):")
    fill_cases = gen.generate_by_operation("FILL", count=5)
    for case in fill_cases[:3]:
        print(f"  Before: {case.before_latex[:50]}  (has empty {{}})")
        print(f"  After:  {case.after_latex[:50]}")
        print(f"  Desc:   {case.edit_description}")
        print()
    
    # Show edit positions
    print("\n[D] Edit position tracking:")
    from synthetic_case_generator import SingleSymbolCaseGenerator
    single_cases = SingleSymbolCaseGenerator.generate_add_single_symbol(count=5)
    
    for case in single_cases[:3]:
        before = case.before_latex
        after = case.after_latex
        # Find where the edit happened
        if before:
            for i, (a, b) in enumerate(zip(before, after)):
                if a != b:
                    print(f"  Edit at position {i}: '{before}' -> '{after}'")
                    break
        else:
            print(f"  Initial: '' -> '{after}'")
    
    return replace_cases + add_cases + fill_cases


# =============================================================================
# 5. HANDWRITING STROKES (Complete vs Incomplete)
# =============================================================================

def demo_handwriting_strokes():
    """Demonstrate handwriting stroke data with complete/incomplete cases."""
    print("\n" + "=" * 70)
    print("5. HANDWRITING STROKES (Complete vs Incomplete)")
    print("=" * 70)
    
    try:
        from scripts.stroke_renderer import (
            StrokeDataLoader,
            StrokeRenderer,
            AugmentationConfig,
            get_available_symbols,
            get_symbol_info,
        )
        from scripts.stroke_dataset import (
            StrokeDatasetGenerator,
            StrokeExample,
            get_symbol_stroke_count,
        )
    except ImportError as e:
        print(f"  [!] Stroke modules not available: {e}")
        return None
    
    # Load stroke data
    print("\n[A] Available handwriting stroke data:")
    try:
        symbols = get_available_symbols()
        print(f"  Total symbols with stroke data: {len(symbols)}")
        print(f"  Sample symbols: {symbols[:20]}...")
    except Exception as e:
        print(f"  [!] Could not load stroke data: {e}")
        print("  Showing conceptual example instead...")
        demo_stroke_concept()
        return None
    
    # Show symbol info
    print("\n[B] Symbol stroke information:")
    for sym in ['α', 'β', 'x', '+', '∫'][:5]:
        info = get_symbol_info(sym)
        if info:
            print(f"  {sym}: {info['num_strokes']} strokes, "
                  f"{info['total_variations']} variations")
    
    # Generate stroke progressions
    print("\n[C] Stroke progression (Complete vs Incomplete cases):")
    try:
        generator = StrokeDatasetGenerator(seed=42)
        
        # Pick a multi-stroke symbol
        test_symbol = 'α'  # alpha typically has 1-2 strokes
        sym_info = get_symbol_info(test_symbol)
        
        if sym_info:
            total_strokes = sym_info['num_strokes']
            print(f"\n  Symbol: {test_symbol} ({sym_info['latex']})")
            print(f"  Total strokes: {total_strokes}")
            
            # Generate all stroke stages
            examples = generator.generate_stroke_progression(
                symbol=test_symbol,
                context=r"x^2 + ",  # Previous context
                include_all=True,   # All intermediate stages
            )
            
            print(f"\n  Generated {len(examples)} stroke stages:")
            for ex in examples:
                status = "COMPLETE" if ex.is_complete else "INCOMPLETE"
                print(f"    Stroke {ex.stroke_idx + 1}/{ex.total_strokes}: "
                      f"output='{ex.output}' [{status}]")
                print(f"      Canvas shape: {ex.canvas.shape}")
    except Exception as e:
        print(f"  [!] Could not generate stroke progression: {e}")
    
    # Training format for complete vs incomplete
    print("\n[D] Training format comparison:")
    print("""
    INCOMPLETE (waiting for more strokes):
        Input:  [canvas with 1 of 3 strokes]
        Context: "x^2 + "
        Output: "<WAIT>"   <- Model learns to wait
        
    COMPLETE (all strokes given):
        Input:  [canvas with 3 of 3 strokes]  
        Context: "x^2 + "
        Output: "\\alpha"  <- Model outputs the symbol
    """)
    
    return symbols


def demo_stroke_concept():
    """Show conceptual stroke handling when data not available."""
    print("""
    Conceptual Stroke Handling:
    ==========================
    
    For a symbol like 'α' (alpha) with 2 strokes:
    
    INCOMPLETE (stroke 1 of 2):
    +------------------+
    |     \\           |   <- First stroke drawn
    |      \\          |
    |       \\         |
    +------------------+
    Model output: <WAIT>
    
    COMPLETE (stroke 2 of 2):
    +------------------+
    |     /\\          |   <- Both strokes
    |    /  \\         |
    |   /    \\        |
    +------------------+  
    Model output: \\alpha
    
    Training Examples:
    -----------------
    1. {"canvas": [partial], "context": "x + ", "output": "<WAIT>"}
    2. {"canvas": [complete], "context": "x + ", "output": "\\alpha"}
    """)


# =============================================================================
# 6. FULL TRAINING EXAMPLE FORMAT
# =============================================================================

def demo_training_format():
    """Show the full training example format."""
    print("\n" + "=" * 70)
    print("6. FULL TRAINING EXAMPLE FORMAT")
    print("=" * 70)
    
    print("""
    Training Example Structure:
    ==========================
    
    {
        "id": "example_0001",
        
        # INPUT (what model sees)
        "latex_context": "x^2 + y^2 = ",      # Previous LaTeX
        "edit_region_image": [H, W, C],        # Crop of handwriting area
        "edit_start_pos": 12,                  # Where edit starts in context
        "edit_end_pos": 12,                    # Where old content ends (same = ADD)
        
        # Stroke info (for real-time)
        "stroke_canvas": [256, 256],           # Rendered strokes
        "stroke_count": 2,                     # How many strokes given
        "is_complete": true,                   # All strokes provided?
        
        # OUTPUT (autoregressive target)
        "output": "r^2",                       # What model should produce
        
        # METADATA
        "depth": 2,
        "operation": "ADD",
        "category": "algebra",
    }
    
    At inference, the result is:
        context[:start] + output + context[end:]
        = "x^2 + y^2 = " + "r^2" + ""
        = "x^2 + y^2 = r^2"
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all demos."""
    print("=" * 70)
    print("FULL PIPELINE DEMO: LaTeX Editing Training Data Generation")
    print("=" * 70)
    
    # 1. Generate atomic samples
    atomic_cases = demo_atomic_generation()
    
    # 2. Compositional augmentation
    composed = demo_compositional_augmentation(atomic_cases)
    
    # 3. Render LaTeX to images
    rendered = demo_latex_rendering(atomic_cases)
    
    # 4. Edit symbols
    edit_cases = demo_symbol_editing()
    
    # 5. Handwriting strokes
    stroke_data = demo_handwriting_strokes()
    
    # 6. Training format
    demo_training_format()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    
    # Summary
    print(f"""
    Summary:
    --------
    - Generated {len(atomic_cases)} atomic cases at specific depths
    - Demonstrated compositional context building
    - {'Rendered' if rendered else 'Skipped rendering'} LaTeX to images
    - Generated edit cases (REPLACE, ADD, FILL)
    - {'Loaded' if stroke_data else 'Conceptualized'} handwriting stroke data
    
    Key APIs:
    ---------
    gen = CaseGenerator(seed=42)
    cases = gen.generate_at_depth(target_depth=3, count=100)
    cases = gen.generate_curriculum(depth_distribution={{1: 0.3, 2: 0.3, 3: 0.4}})
    
    from scripts.latex_renderer import render_latex
    img = render_latex(r"x^2 + y^2", dpi=150)
    
    from scripts.stroke_dataset import StrokeDatasetGenerator
    gen = StrokeDatasetGenerator()
    examples = gen.generate_stroke_progression(symbol='α', context='x + ')
    """)


if __name__ == "__main__":
    main()

