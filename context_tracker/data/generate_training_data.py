"""
Unified Training Data Generator

Simple interface to generate all training data types:
- Atomic samples at controlled depth
- Commutative diagrams with labels
- Compositional context (metadata only, no full images)
- Stroke progressions (complete vs incomplete)
- Position embeddings (stroke center with jitter)
- Distractor strokes

Usage:
    # Quick generation
    from context_tracker.data.generate_training_data import generate_all
    data = generate_all(count=1000, seed=42)
    
    # Or step by step
    from context_tracker.data.generate_training_data import (
        generate_atomic_cases,
        generate_compositional_context,
        generate_stroke_examples,
        generate_commutative_diagrams,
    )
    
    cases = generate_atomic_cases(depth=3, count=100)
    contexts = generate_compositional_context(count=500, separator=",")
    strokes = generate_stroke_examples(symbols=['x', 'y'], include_incomplete=True)
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

# Ensure imports work
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from synthetic_case_generator import CaseGenerator, analyze_latex_complexity, EditCase
from augmentation import ChunkPool, ContextComposer, ComposedTrainingExample
from scripts.stroke_dataset import StrokeDatasetGenerator
from scripts.stroke_renderer import StrokeDataLoader, StrokeRenderer


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SymbolPosition:
    """Position info for a symbol with stroke-based and latex-based coordinates."""
    symbol: str
    latex: str
    
    # Stroke position (from handwriting, normalized 0-1)
    stroke_center: Tuple[float, float]
    stroke_bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    
    # LaTeX position (from rendering, normalized 0-1)  
    latex_center: Optional[Tuple[float, float]] = None
    latex_bbox: Optional[Tuple[float, float, float, float]] = None
    
    # Position offset for embedding (stroke - latex)
    position_offset: Optional[Tuple[float, float]] = None
    
    # Jitter applied to simulate handwriting
    jitter: Tuple[float, float] = (0.0, 0.0)


@dataclass
class StrokeState:
    """Stroke state for a symbol (complete or incomplete)."""
    symbol: str
    latex: str
    strokes_drawn: int
    total_strokes: int
    is_complete: bool
    expected_output: str  # The latex if complete, "<WAIT>" if incomplete
    position: SymbolPosition
    canvas: Optional[np.ndarray] = None
    
    
@dataclass
class DistractorInfo:
    """Info about a distractor stroke (incomplete symbol as noise)."""
    symbol: str
    latex: str
    strokes_drawn: int
    total_strokes: int
    position_offset: Tuple[float, float]


@dataclass
class TrainingCase:
    """
    Unified training case format.
    
    Contains all info needed for model training:
    - LaTeX context (metadata, no full image needed)
    - Target symbol with stroke state
    - Position embedding info
    - Optional distractors
    """
    id: str
    case_type: str  # 'atomic', 'compositional', 'diagram'
    
    # LaTeX info
    latex_before: str
    latex_after: str
    operation: str
    depth: int
    
    # Edit info
    edit_position: int
    edit_old: str
    edit_new: str
    
    # Symbol positions (each symbol in context)
    symbol_positions: List[Dict[str, Any]]
    
    # Target stroke info
    target: StrokeState
    
    # Optional distractors
    distractors: List[DistractorInfo] = field(default_factory=list)
    
    # Metadata
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Core Generation Functions
# =============================================================================

def generate_atomic_cases(
    depth: int = None,
    min_depth: int = 1,
    max_depth: int = 4,
    count: int = 100,
    seed: int = None,
) -> List[EditCase]:
    """
    Generate atomic edit cases at controlled depth.
    
    Args:
        depth: Exact depth (if set, ignores min/max)
        min_depth: Minimum depth
        max_depth: Maximum depth  
        count: Number of cases
        seed: Random seed
        
    Returns:
        List of EditCase
        
    Example:
        cases = generate_atomic_cases(depth=3, count=50)
        cases = generate_atomic_cases(min_depth=1, max_depth=4, count=200)
    """
    gen = CaseGenerator(seed=seed)
    
    if depth is not None:
        return gen.generate_at_depth(target_depth=depth, count=count)
    else:
        return gen.generate_by_depth_range(min_depth=min_depth, max_depth=max_depth, count=count)


def generate_commutative_diagrams(
    count: int = 20,
    include_labels: bool = True,
    mixed_case: bool = True,
    seed: int = None,
) -> List[Dict[str, Any]]:
    """
    Generate commutative diagram LaTeX with metadata.
    
    Args:
        count: Number of diagrams
        include_labels: Include arrow labels (f, g, etc.)
        mixed_case: Mix upper/lower case node labels
        seed: Random seed
        
    Returns:
        List of diagram dicts with 'latex' and 'nodes' keys
        
    Example:
        diagrams = generate_commutative_diagrams(count=10)
    """
    if seed is not None:
        random.seed(seed)
    
    # Node pools
    upper = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lower = list("abcdefghijklmnopqrstuvwxyz")
    greek = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\phi", "\\psi"]
    labels = list("fghkmnpqrst") + greek
    
    diagrams = []
    
    templates = [
        # Simple arrow A -> B
        lambda n, l: (
            f"\\begin{{tikzcd}} {n[0]} \\arrow[r, \"{l[0]}\"] & {n[1]} \\end{{tikzcd}}",
            [n[0], n[1]], [l[0]]
        ),
        # Chain A -> B -> C
        lambda n, l: (
            f"\\begin{{tikzcd}} {n[0]} \\arrow[r, \"{l[0]}\"] & {n[1]} \\arrow[r, \"{l[1]}\"] & {n[2]} \\end{{tikzcd}}",
            [n[0], n[1], n[2]], [l[0], l[1]]
        ),
        # Triangle
        lambda n, l: (
            f"\\begin{{tikzcd}} {n[0]} \\arrow[r, \"{l[0]}\"] \\arrow[dr, \"{l[2]}\"'] & {n[1]} \\arrow[d, \"{l[1]}\"] \\\\ & {n[2]} \\end{{tikzcd}}",
            [n[0], n[1], n[2]], [l[0], l[1], l[2]]
        ),
        # Square
        lambda n, l: (
            f"\\begin{{tikzcd}} {n[0]} \\arrow[r, \"{l[0]}\"] \\arrow[d, \"{l[2]}\"'] & {n[1]} \\arrow[d, \"{l[1]}\"] \\\\ {n[2]} \\arrow[r, \"{l[3]}\"'] & {n[3]} \\end{{tikzcd}}",
            [n[0], n[1], n[2], n[3]], [l[0], l[1], l[2], l[3]]
        ),
        # Long chain A -> B -> C -> D
        lambda n, l: (
            f"\\begin{{tikzcd}} {n[0]} \\arrow[r, \"{l[0]}\"] & {n[1]} \\arrow[r, \"{l[1]}\"] & {n[2]} \\arrow[r, \"{l[2]}\"] & {n[3]} \\end{{tikzcd}}",
            [n[0], n[1], n[2], n[3]], [l[0], l[1], l[2]]
        ),
    ]
    
    for i in range(count):
        # Select nodes
        if mixed_case:
            pool = upper + lower
        else:
            pool = upper
        nodes = random.sample(pool, 4)
        arrow_labels = random.sample(labels, 4) if include_labels else [""] * 4
        
        # Pick template
        template = random.choice(templates)
        latex, used_nodes, used_labels = template(nodes, arrow_labels)
        
        # Create node position metadata (normalized)
        node_positions = []
        for j, node in enumerate(used_nodes):
            # Approximate position based on typical tikzcd layout
            col = j % 2
            row = j // 2
            node_positions.append({
                'symbol': node,
                'latex_center': (0.3 + col * 0.4, 0.3 + row * 0.4),
                'latex_bbox': (0.2 + col * 0.4, 0.2 + row * 0.4, 
                              0.4 + col * 0.4, 0.4 + row * 0.4),
            })
        
        diagrams.append({
            'id': f'diagram_{i:03d}',
            'latex': latex,
            'nodes': used_nodes,
            'arrow_labels': used_labels if include_labels else [],
            'node_positions': node_positions,
            'type': 'commutative_diagram',
        })
    
    return diagrams


def generate_compositional_context(
    count: int = 100,
    chunks_per_depth: int = 50,
    max_depth: int = 4,
    num_context_chunks: int = 3,
    separator: str = ", ",  # Natural separator
    seed: int = None,
    add_position_jitter: bool = True,
    jitter_range: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Generate compositional context (metadata only, no full images).
    
    The model only sees where strokes are drawn, so we don't need
    to render the full composed expression. We just need:
    - LaTeX string with symbol positions
    - Edit location and target
    - Position embedding info
    
    Args:
        count: Number of examples
        chunks_per_depth: Chunks per depth level for pool
        max_depth: Maximum depth
        num_context_chunks: Context chunks per example
        separator: Natural separator between chunks (", " recommended)
        seed: Random seed
        add_position_jitter: Add random jitter to positions
        jitter_range: Jitter range (normalized 0-1)
        
    Returns:
        List of compositional context dicts
        
    Example:
        contexts = generate_compositional_context(count=500, separator=", ")
    """
    gen = CaseGenerator(seed=seed)
    pool = ChunkPool.from_case_generator(gen, chunks_per_depth=chunks_per_depth, max_depth=max_depth)
    composer = ContextComposer(pool, seed=seed)
    
    # Generate examples
    examples = composer.compose_batch(
        num_examples=count,
        num_context_chunks=num_context_chunks,
        separator=separator,
    )
    
    result = []
    for i, ex in enumerate(examples):
        # Extract symbol positions from chunks
        symbol_positions = []
        current_pos = 0
        
        for chunk in ex.context_chunks:
            chunk_len = len(chunk.latex)
            # Approximate center position in full string
            rel_pos = (current_pos + chunk_len / 2) / len(ex.full_context_before)
            
            # Base position (normalized 0-1)
            base_center = (rel_pos, 0.5)
            base_bbox = (rel_pos - 0.1, 0.3, rel_pos + 0.1, 0.7)
            
            # Add jitter if enabled
            if add_position_jitter:
                jitter = (random.uniform(-jitter_range, jitter_range),
                         random.uniform(-jitter_range, jitter_range))
            else:
                jitter = (0, 0)
            
            jittered_center = (base_center[0] + jitter[0], base_center[1] + jitter[1])
            
            symbol_positions.append({
                'chunk_latex': chunk.latex,
                'depth': chunk.depth,
                'latex_center': base_center,
                'latex_bbox': base_bbox,
                'stroke_center': jittered_center,  # With jitter
                'position_offset': jitter,
            })
            
            current_pos += chunk_len + len(separator)
        
        result.append({
            'id': f'comp_{i:04d}',
            'full_context_before': ex.full_context_before,
            'full_context_after': ex.full_context_after,
            'edit_start_pos': ex.edit_start_pos,
            'edit_end_pos': ex.edit_end_pos,
            'edit_old': ex.edit_old_content,
            'edit_new': ex.edit_new_content,
            'context_depths': ex.context_depths,
            'target_depth': ex.target_depth,
            'symbol_positions': symbol_positions,
            'separator': separator,
            'type': 'compositional',
        })
    
    return result


def generate_stroke_examples(
    symbols: List[str] = None,
    count_per_symbol: int = 5,
    include_incomplete: bool = True,
    add_distractors: bool = True,
    num_distractors: int = 3,
    add_position_jitter: bool = True,
    jitter_range: float = 0.1,
    seed: int = None,
) -> List[Dict[str, Any]]:
    """
    Generate stroke examples with complete/incomplete states.
    
    Args:
        symbols: Specific symbols (None = use all available)
        count_per_symbol: Examples per symbol
        include_incomplete: Include incomplete stroke states
        add_distractors: Add distractor strokes around target
        num_distractors: Number of distractors per example
        add_position_jitter: Add jitter to stroke positions
        jitter_range: Jitter range (normalized)
        seed: Random seed
        
    Returns:
        List of stroke example dicts
        
    Example:
        examples = generate_stroke_examples(symbols=['x', 'y', '+'], include_incomplete=True)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    stroke_gen = StrokeDatasetGenerator(seed=seed)
    loader = stroke_gen.loader
    available = loader.get_all_symbols()
    
    if symbols is None:
        # Use symbols with 2+ strokes for interesting progression
        symbols = []
        for s in available[:50]:
            sym_data = loader.get_symbol(s)
            if sym_data and sym_data.num_strokes >= 2:
                symbols.append(s)
        symbols = symbols[:10]  # Limit
    else:
        symbols = [s for s in symbols if s in available]
    
    results = []
    
    for symbol in symbols:
        sym_data = loader.get_symbol(symbol)
        if sym_data is None:
            continue
        
        for case_idx in range(count_per_symbol):
            # Generate progression
            progressions = stroke_gen.generate_stroke_progression(
                symbol=symbol,
                context="",
                include_all=include_incomplete
            )
            
            for prog in progressions:
                # Add position jitter
                if add_position_jitter:
                    jitter = (random.uniform(-jitter_range, jitter_range),
                             random.uniform(-jitter_range, jitter_range))
                else:
                    jitter = (0, 0)
                
                # Calculate stroke center from canvas
                if prog.stroke_points:
                    xs = [p[0] for p in prog.stroke_points]
                    ys = [p[1] for p in prog.stroke_points]
                    stroke_center = (np.mean(xs) / 256, np.mean(ys) / 256)  # Normalize
                else:
                    stroke_center = (0.5, 0.5)
                
                # Apply jitter
                jittered_center = (stroke_center[0] + jitter[0], stroke_center[1] + jitter[1])
                
                example = {
                    'id': f'{symbol}_{case_idx}_{prog.stroke_idx+1}of{prog.total_strokes}',
                    'symbol': symbol,
                    'latex': sym_data.latex,
                    'strokes_drawn': prog.stroke_idx + 1,
                    'total_strokes': prog.total_strokes,
                    'is_complete': prog.is_complete,
                    'expected_output': prog.output,
                    'stroke_center': jittered_center,
                    'latex_center': (0.5, 0.5),
                    'position_offset': jitter,
                    'variation_indices': prog.variation_indices,
                    'type': 'stroke_progression',
                }
                
                # Add distractors if complete and enabled
                if prog.is_complete and add_distractors:
                    distractors = []
                    for _ in range(num_distractors):
                        dist_sym = random.choice(available)
                        dist_data = loader.get_symbol(dist_sym)
                        if dist_data and dist_data.num_strokes > 1:
                            # Incomplete distractor
                            dist_strokes = random.randint(1, dist_data.num_strokes - 1)
                            angle = random.uniform(0, 2 * np.pi)
                            dist = random.uniform(0.15, 0.35)
                            
                            distractors.append({
                                'symbol': dist_sym,
                                'latex': dist_data.latex,
                                'strokes_drawn': dist_strokes,
                                'total_strokes': dist_data.num_strokes,
                                'is_complete': False,
                                'position_offset': (dist * np.cos(angle), dist * np.sin(angle)),
                            })
                    
                    example['distractors'] = distractors
                
                results.append(example)
    
    return results


# =============================================================================
# Unified Generation
# =============================================================================

def generate_all(
    atomic_count: int = 200,
    compositional_count: int = 500,
    diagram_count: int = 50,
    stroke_count_per_symbol: int = 3,
    seed: int = 42,
    separator: str = ", ",
    include_incomplete: bool = True,
    add_distractors: bool = True,
    add_position_jitter: bool = True,
) -> Dict[str, List]:
    """
    Generate complete training dataset with all case types.
    
    Args:
        atomic_count: Atomic cases (split across depths 1-4)
        compositional_count: Compositional context examples
        diagram_count: Commutative diagrams
        stroke_count_per_symbol: Stroke examples per symbol
        seed: Random seed
        separator: Separator for compositional context
        include_incomplete: Include incomplete stroke states
        add_distractors: Add distractor strokes
        add_position_jitter: Add position jitter
        
    Returns:
        Dict with keys: 'atomic', 'compositional', 'diagrams', 'strokes'
        
    Example:
        data = generate_all(atomic_count=500, compositional_count=2000)
        
        # Access specific data
        for case in data['atomic']:
            print(case.before_latex, case.after_latex)
            
        for ctx in data['compositional']:
            print(ctx['full_context_before'], ctx['symbol_positions'])
    """
    print("=" * 60)
    print("Generating Training Data")
    print("=" * 60)
    
    # Atomic cases at each depth
    print(f"\n1. Atomic Cases ({atomic_count} total)")
    atomic_per_depth = atomic_count // 4
    atomic = []
    for depth in [1, 2, 3, 4]:
        cases = generate_atomic_cases(depth=depth, count=atomic_per_depth, seed=seed)
        atomic.extend(cases)
        print(f"   Depth {depth}: {len(cases)} cases")
    
    # Compositional context (metadata only)
    print(f"\n2. Compositional Context ({compositional_count} examples)")
    compositional = generate_compositional_context(
        count=compositional_count,
        separator=separator,
        seed=seed,
        add_position_jitter=add_position_jitter,
    )
    print(f"   Generated: {len(compositional)} contexts")
    
    # Commutative diagrams
    print(f"\n3. Commutative Diagrams ({diagram_count})")
    diagrams = generate_commutative_diagrams(
        count=diagram_count,
        include_labels=True,
        mixed_case=True,
        seed=seed,
    )
    print(f"   Generated: {len(diagrams)} diagrams")
    
    # Stroke examples
    print(f"\n4. Stroke Examples")
    strokes = generate_stroke_examples(
        count_per_symbol=stroke_count_per_symbol,
        include_incomplete=include_incomplete,
        add_distractors=add_distractors,
        add_position_jitter=add_position_jitter,
        seed=seed,
    )
    print(f"   Generated: {len(strokes)} stroke examples")
    
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print(f"  Atomic: {len(atomic)}")
    print(f"  Compositional: {len(compositional)}")
    print(f"  Diagrams: {len(diagrams)}")
    print(f"  Strokes: {len(strokes)}")
    print("=" * 60)
    
    return {
        'atomic': atomic,
        'compositional': compositional,
        'diagrams': diagrams,
        'strokes': strokes,
    }


def save_training_data(data: Dict[str, List], output_dir: str):
    """
    Save generated training data to JSON files.
    
    Args:
        data: Output from generate_all()
        output_dir: Output directory
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Atomic (convert EditCase to dict)
    atomic_data = []
    for case in data['atomic']:
        atomic_data.append({
            'id': case.id,
            'operation': case.operation,
            'category': case.category,
            'before_latex': case.before_latex,
            'after_latex': case.after_latex,
            'depth': case.depth,
        })
    
    with open(out / 'atomic_cases.json', 'w', encoding='utf-8') as f:
        json.dump(atomic_data, f, indent=2, ensure_ascii=False)
    
    # Compositional
    with open(out / 'compositional_context.json', 'w', encoding='utf-8') as f:
        json.dump(data['compositional'], f, indent=2, ensure_ascii=False)
    
    # Diagrams
    with open(out / 'commutative_diagrams.json', 'w', encoding='utf-8') as f:
        json.dump(data['diagrams'], f, indent=2, ensure_ascii=False)
    
    # Strokes
    with open(out / 'stroke_examples.json', 'w', encoding='utf-8') as f:
        json.dump(data['strokes'], f, indent=2, ensure_ascii=False)
    
    print(f"Saved training data to {output_dir}/")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Generate training data from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--output", "-o", default="training_data", help="Output directory")
    parser.add_argument("--atomic", type=int, default=200, help="Atomic cases")
    parser.add_argument("--compositional", type=int, default=500, help="Compositional contexts")
    parser.add_argument("--diagrams", type=int, default=50, help="Commutative diagrams")
    parser.add_argument("--strokes", type=int, default=3, help="Stroke examples per symbol")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--separator", default=", ", help="Context separator")
    
    args = parser.parse_args()
    
    data = generate_all(
        atomic_count=args.atomic,
        compositional_count=args.compositional,
        diagram_count=args.diagrams,
        stroke_count_per_symbol=args.strokes,
        seed=args.seed,
        separator=args.separator,
    )
    
    save_training_data(data, args.output)


if __name__ == "__main__":
    main()

