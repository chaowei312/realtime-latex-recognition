"""
Stroke-Level Training Dataset Generator

Combines:
- Captured stroke data (stroke_data_merged.json)
- Edit cases (cases.py)
- GPU stroke rendering

Generates training examples with:
- Rendered handwritten symbols
- LaTeX context
- Edit operations (ADD, REPLACE, INSERT, WAIT)

Usage:
    from scripts.stroke_dataset import StrokeDatasetGenerator
    
    gen = StrokeDatasetGenerator()
    examples = gen.generate(count=1000)
"""

import random
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .stroke_renderer import (
    StrokeDataLoader,
    StrokeRenderer, 
    GPUStrokeRenderer,
    AugmentationConfig,
    get_symbol_info,
    render_symbol_group,
)

# Try importing synthetic_case_generator.py
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from synthetic_case_generator import CaseGenerator, EditCase
    HAS_CASES = True
except ImportError:
    HAS_CASES = False
    print("Warning: synthetic_case_generator.py not found. Using standalone generation.")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class StrokeExample:
    """
    A single stroke-level training example.
    
    For multi-stroke symbols:
        - If incomplete (stroke_idx < total_strokes): output = "<WAIT>"
        - If complete: output = symbol/latex
    """
    id: str
    
    # Symbol info
    symbol: str
    latex: str
    
    # Stroke progress
    stroke_idx: int           # Current stroke (0-indexed)
    total_strokes: int        # Total strokes for symbol
    is_complete: bool         # True if this is the final stroke
    
    # Rendered data
    canvas: np.ndarray        # [H, W] rendered strokes so far
    stroke_points: List[Tuple[float, float]]  # All points for conv encoding
    
    # Context
    latex_context: str        # Previous LaTeX (what's already recognized)
    
    # Output
    output: str               # "<WAIT>" or the symbol/latex
    
    # Metadata
    variation_indices: List[int]  # Which variations were used
    augmentation_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'latex': self.latex,
            'stroke_progress': f"{self.stroke_idx + 1}/{self.total_strokes}",
            'is_complete': self.is_complete,
            'canvas_shape': list(self.canvas.shape),
            'num_points': len(self.stroke_points),
            'latex_context': self.latex_context,
            'output': self.output,
            'variation_indices': self.variation_indices,
        }


@dataclass
class EditExample:
    """
    Training example for an edit operation.
    
    Combines:
    - Before/after LaTeX
    - Rendered edit region (one or more symbols)
    - Operation type
    """
    id: str
    
    # Edit info
    operation: str            # ADD, REPLACE, INSERT, FILL, DELETE
    before_latex: str
    after_latex: str
    edit_symbols: List[str]   # Symbol(s) being edited (can be multiple)
    
    # Rendered edit region
    canvas: np.ndarray        # [H, W] rendered symbol(s)
    symbol_metadata: List[Dict[str, Any]]  # Per-symbol bbox and info
    
    # Metadata
    category: str
    difficulty: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'operation': self.operation,
            'before_latex': self.before_latex,
            'after_latex': self.after_latex,
            'edit_symbols': self.edit_symbols,
            'canvas_shape': list(self.canvas.shape),
            'symbol_metadata': self.symbol_metadata,
            'category': self.category,
            'difficulty': self.difficulty,
        }


@dataclass 
class TrainingExample:
    """
    Complete training example ready for model input.
    
    Structure:
        Input: (stroke_image, latex_context)
        Output: target_token or <WAIT>
    """
    id: str
    
    # Input
    stroke_image: np.ndarray      # [H, W] rendered handwritten strokes
    latex_context: str            # Current LaTeX state before edit
    
    # Output  
    target: str                   # Target token(s) or "<WAIT>"
    
    # Edit metadata
    operation: str                # ADD, REPLACE, etc.
    edit_symbols: List[str]       # What was written
    symbol_bboxes: List[Dict]     # Per-symbol bounding boxes in stroke_image
    
    # Training metadata
    is_complete: bool             # True if all strokes for symbol(s) drawn
    category: str
    difficulty: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'stroke_image_shape': list(self.stroke_image.shape),
            'latex_context': self.latex_context,
            'target': self.target,
            'operation': self.operation,
            'edit_symbols': self.edit_symbols,
            'symbol_bboxes': self.symbol_bboxes,
            'is_complete': self.is_complete,
            'category': self.category,
            'difficulty': self.difficulty,
        }


# =============================================================================
# Symbol Stroke Count Database
# =============================================================================

def get_symbol_stroke_count(symbol: str, loader: StrokeDataLoader) -> int:
    """
    Get the expected number of strokes for a symbol.
    Uses the captured data's stroke structure.
    """
    sym_data = loader.get_symbol(symbol)
    if sym_data:
        return sym_data.num_strokes
    
    # Fallback defaults for common symbols not in captured data
    DEFAULTS = {
        # Single stroke
        'a': 1, 'c': 1, 'e': 1, 'o': 1, 's': 1, 'u': 1, 'v': 1, 'w': 1,
        'C': 1, 'O': 1, 'S': 1, 'U': 1, 'V': 1, 'W': 1,
        '(': 1, ')': 1, '[': 1, ']': 1,
        
        # Two strokes
        '+': 2, '=': 2, 'x': 2, 't': 2, 'i': 2, 'j': 2,
        'A': 2, 'T': 2, 'X': 2, 'Y': 2,
        
        # Three strokes
        '8': 3, 'B': 3, 'E': 3, 'F': 3, 'H': 3,
    }
    
    return DEFAULTS.get(symbol, 2)


# =============================================================================
# Dataset Generator
# =============================================================================

class StrokeDatasetGenerator:
    """Generate stroke-level training dataset."""
    
    def __init__(self,
                 data_path: Optional[str] = None,
                 canvas_size: int = 256,
                 augmentation: Optional[AugmentationConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            data_path: Path to stroke_data_merged.json
            canvas_size: Render canvas size
            augmentation: Augmentation config
            seed: Random seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.loader = StrokeDataLoader(data_path)
        self.renderer = StrokeRenderer(
            self.loader, 
            canvas_size=canvas_size,
            augmentation=augmentation or AugmentationConfig()
        )
        self.canvas_size = canvas_size
        self.available_symbols = self.loader.get_all_symbols()
    
    def generate_stroke_progression(self,
                                    symbol: str,
                                    context: str = "",
                                    include_all: bool = True) -> List[StrokeExample]:
        """
        Generate training examples for all stroke stages of a symbol.
        
        For a 3-stroke symbol, generates:
        - After stroke 1: output = <WAIT>
        - After stroke 2: output = <WAIT>
        - After stroke 3: output = symbol
        
        Args:
            symbol: Symbol to generate
            context: Previous LaTeX context
            include_all: Include all intermediate stages, or just final
            
        Returns:
            List of StrokeExample
        """
        sym_data = self.loader.get_symbol(symbol)
        if sym_data is None:
            return []
        
        examples = []
        stroke_names = sorted(sym_data.strokes.keys())
        total_strokes = len(stroke_names)
        
        # Select random variations
        variation_indices = [
            random.randint(0, len(sym_data.strokes[name]) - 1)
            for name in stroke_names
        ]
        
        # Sample augmentation (same for all stages of this symbol)
        aug_params = self.renderer.augmentation.sample()
        
        if include_all:
            stroke_range = range(1, total_strokes + 1)
        else:
            stroke_range = [total_strokes]
        
        for num_strokes in stroke_range:
            is_complete = (num_strokes >= total_strokes)
            
            # Render with only first num_strokes
            canvas, points = self._render_partial(
                sym_data, 
                stroke_names[:num_strokes],
                variation_indices[:num_strokes],
                aug_params
            )
            
            example = StrokeExample(
                id=f"{symbol}_{num_strokes}of{total_strokes}",
                symbol=sym_data.symbol,
                latex=sym_data.latex,
                stroke_idx=num_strokes - 1,
                total_strokes=total_strokes,
                is_complete=is_complete,
                canvas=canvas,
                stroke_points=points,
                latex_context=context,
                output=sym_data.latex if is_complete else "<WAIT>",
                variation_indices=variation_indices[:num_strokes],
                augmentation_params=aug_params,
            )
            examples.append(example)
        
        return examples
    
    def _render_partial(self,
                       sym_data,
                       stroke_names: List[str],
                       variation_indices: List[int],
                       aug_params: Dict[str, Any]) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Render only specified strokes."""
        from .stroke_renderer import StrokeData
        
        all_points = []
        
        # Collect strokes to render
        strokes_to_render = []
        for i, stroke_name in enumerate(stroke_names):
            var_idx = variation_indices[i]
            stroke = sym_data.strokes[stroke_name][var_idx]
            
            # Apply augmentation
            stroke = stroke.interpolate(self.renderer.augmentation.interpolate_points)
            stroke = stroke.apply_transform(
                rotation=aug_params['rotation'],
                scale=aug_params['scale'],
                translate=aug_params['translate']
            )
            if aug_params['jitter'] > 0:
                stroke = stroke.add_jitter(aug_params['jitter'])
            
            strokes_to_render.append(stroke)
            
            for p in stroke.points:
                all_points.append((p.x * self.canvas_size, p.y * self.canvas_size))
        
        # Render
        canvas = self.renderer._render_strokes(strokes_to_render, aug_params['thickness'])
        
        return canvas, all_points
    
    def generate_random_examples(self, count: int = 100) -> List[StrokeExample]:
        """
        Generate random stroke examples from available symbols.
        
        Args:
            count: Number of examples to generate
            
        Returns:
            List of StrokeExample
        """
        examples = []
        
        for i in range(count):
            symbol = random.choice(self.available_symbols)
            
            # Random context (sometimes empty)
            if random.random() < 0.3:
                context = ""
            else:
                context_options = ["x + ", "y = ", r"\frac{", "f(x) = "]
                context = random.choice(context_options)
            
            # Generate progression (sometimes all stages, sometimes just final)
            include_all = random.random() < 0.6
            symbol_examples = self.generate_stroke_progression(
                symbol, context, include_all
            )
            
            examples.extend(symbol_examples)
        
        return examples
    
    def generate_with_cases(self, 
                           count: int = 100,
                           multi_symbol_ratio: float = 0.3) -> List[EditExample]:
        """
        Generate examples using edit cases from cases.py.
        
        Args:
            count: Number of examples
            multi_symbol_ratio: Ratio of multi-symbol edits (e.g., "x+y" written together)
            
        Returns:
            List of EditExample
        """
        if not HAS_CASES:
            print("Warning: cases.py not available, using random generation")
            return []
        
        examples = []
        
        case_gen = CaseGenerator()
        cases = case_gen.generate_all(count_per_category=max(1, count // 10))
        
        for case in cases[:count]:
            # Determine edit symbols
            edit_symbols = self._extract_edit_symbols(case)
            
            # Filter to available symbols
            edit_symbols = [s for s in edit_symbols if s in self.available_symbols]
            if not edit_symbols:
                edit_symbols = [random.choice(self.available_symbols)]
            
            # Randomly decide if this is multi-symbol edit
            if len(edit_symbols) == 1 and random.random() < multi_symbol_ratio:
                # Add extra symbols for multi-symbol edit simulation
                extra_count = random.randint(1, 3)
                extra_symbols = random.choices(self.available_symbols, k=extra_count)
                edit_symbols = edit_symbols + extra_symbols
            
            # Render the edit region
            canvas, symbol_metadata = self.renderer.render_symbol_group(
                edit_symbols,
                spacing=random.uniform(0.1, 0.2),
                augment=True,
                return_metadata=True
            )
            
            example = EditExample(
                id=case.id,
                operation=case.operation,
                before_latex=case.before_latex,
                after_latex=case.after_latex,
                edit_symbols=edit_symbols,
                canvas=canvas,
                symbol_metadata=symbol_metadata,
                category=case.category,
                difficulty=case.difficulty,
            )
            examples.append(example)
        
        return examples
    
    def _extract_edit_symbols(self, case) -> List[str]:
        """Extract the symbols being edited from an edit case."""
        symbols = []
        
        # Try metadata first
        if hasattr(case, 'metadata') and case.metadata:
            edit_sym = case.metadata.get('edit_symbol', '')
            if edit_sym:
                symbols.append(edit_sym)
        
        # If no symbols found, try to infer from before/after
        if not symbols:
            # Simple heuristic: find difference between before and after
            before = case.before_latex if hasattr(case, 'before_latex') else ''
            after = case.after_latex if hasattr(case, 'after_latex') else ''
            
            # For ADD operations, new content is at the end
            if case.operation == 'ADD' and len(after) > len(before):
                new_part = after[len(before):]
                for char in new_part:
                    if char in self.available_symbols:
                        symbols.append(char)
            
            # For other operations, just grab recognizable symbols
            if not symbols:
                for char in after:
                    if char in self.available_symbols and char not in symbols:
                        symbols.append(char)
                        if len(symbols) >= 3:
                            break
        
        return symbols if symbols else ['x']  # Fallback
    
    def generate_training_examples(self,
                                   count: int = 1000,
                                   wait_ratio: float = 0.4,
                                   multi_symbol_ratio: float = 0.2) -> List[TrainingExample]:
        """
        Generate complete training examples ready for model input.
        
        Args:
            count: Total number of examples
            wait_ratio: Ratio of incomplete (WAIT) examples
            multi_symbol_ratio: Ratio of multi-symbol edit examples
            
        Returns:
            List of TrainingExample ready for training
        """
        examples = []
        
        if HAS_CASES:
            # Use cases.py for realistic edit scenarios
            case_gen = CaseGenerator()
            cases = case_gen.generate_all(count_per_category=max(1, count // 10))
            random.shuffle(cases)
        else:
            cases = []
        
        for i in range(count):
            example_id = f"train_{i:06d}"
            
            # Get case or generate random scenario
            if cases and i < len(cases):
                case = cases[i]
                operation = case.operation
                before_latex = case.before_latex
                after_latex = case.after_latex
                category = case.category
                difficulty = case.difficulty
                edit_symbols = self._extract_edit_symbols(case)
            else:
                # Random generation
                operation = random.choice(['ADD', 'REPLACE', 'INSERT'])
                before_latex = random.choice(['x + ', 'y = ', '', 'f(x) = '])
                edit_symbols = random.choices(self.available_symbols, 
                                             k=random.randint(1, 3) if random.random() < multi_symbol_ratio else 1)
                after_latex = before_latex + ''.join(edit_symbols)
                category = 'random'
                difficulty = 'easy'
            
            # Filter to available symbols
            edit_symbols = [s for s in edit_symbols if s in self.available_symbols]
            if not edit_symbols:
                edit_symbols = [random.choice(self.available_symbols)]
            
            # Decide if complete or WAIT
            is_complete = random.random() > wait_ratio
            
            if is_complete:
                # Render complete symbol(s)
                if len(edit_symbols) == 1:
                    canvas, points = self.renderer.render_symbol(
                        edit_symbols[0], augment=True, return_points=True
                    )
                    sym_data = self.loader.get_symbol(edit_symbols[0])
                    symbol_bboxes = [{
                        'symbol': edit_symbols[0],
                        'latex': sym_data.latex if sym_data else edit_symbols[0],
                        'bbox': [0, 0, self.canvas_size, self.canvas_size]
                    }]
                else:
                    canvas, symbol_bboxes = self.renderer.render_symbol_group(
                        edit_symbols, augment=True, return_metadata=True
                    )
                
                # Target is the LaTeX for what was written
                target_parts = []
                for sym in edit_symbols:
                    sym_data = self.loader.get_symbol(sym)
                    target_parts.append(sym_data.latex if sym_data else sym)
                target = ' '.join(target_parts)
            else:
                # Render incomplete (partial strokes for WAIT)
                # Pick one symbol and render partial strokes
                symbol = edit_symbols[0]
                sym_data = self.loader.get_symbol(symbol)
                
                if sym_data and sym_data.num_strokes > 1:
                    # Render partial
                    stroke_names = sorted(sym_data.strokes.keys())
                    num_strokes = random.randint(1, len(stroke_names) - 1)
                    
                    variation_indices = [
                        random.randint(0, len(sym_data.strokes[name]) - 1)
                        for name in stroke_names[:num_strokes]
                    ]
                    aug_params = self.renderer.augmentation.sample()
                    
                    canvas, points = self._render_partial(
                        sym_data, stroke_names[:num_strokes],
                        variation_indices, aug_params
                    )
                else:
                    # Symbol has only 1 stroke, render it but mark as WAIT anyway
                    canvas, points = self.renderer.render_symbol(
                        symbol, augment=True, return_points=True
                    )
                
                symbol_bboxes = [{
                    'symbol': symbol,
                    'latex': sym_data.latex if sym_data else symbol,
                    'bbox': [0, 0, self.canvas_size, self.canvas_size],
                    'partial': True
                }]
                target = '<WAIT>'
            
            example = TrainingExample(
                id=example_id,
                stroke_image=canvas,
                latex_context=before_latex,
                target=target,
                operation=operation,
                edit_symbols=edit_symbols,
                symbol_bboxes=symbol_bboxes,
                is_complete=is_complete,
                category=category,
                difficulty=difficulty,
            )
            examples.append(example)
        
        return examples
    
    def generate_balanced(self,
                         total_count: int = 1000,
                         wait_ratio: float = 0.4) -> List[StrokeExample]:
        """
        Generate balanced dataset with specified WAIT ratio.
        
        Args:
            total_count: Total examples to generate
            wait_ratio: Ratio of <WAIT> examples
            
        Returns:
            List of StrokeExample
        """
        examples = []
        
        wait_count = int(total_count * wait_ratio)
        complete_count = total_count - wait_count
        
        # Generate WAIT examples (incomplete symbols)
        for _ in range(wait_count):
            symbol = random.choice(self.available_symbols)
            sym_data = self.loader.get_symbol(symbol)
            if sym_data is None or sym_data.num_strokes <= 1:
                continue
            
            # Generate partial (not all strokes)
            all_examples = self.generate_stroke_progression(symbol, include_all=True)
            incomplete = [e for e in all_examples if not e.is_complete]
            if incomplete:
                examples.append(random.choice(incomplete))
        
        # Generate complete examples
        for _ in range(complete_count):
            symbol = random.choice(self.available_symbols)
            all_examples = self.generate_stroke_progression(symbol, include_all=False)
            if all_examples:
                examples.append(all_examples[0])
        
        random.shuffle(examples)
        return examples
    
    def save_to_json(self, 
                     examples,  # List of any example type
                     filepath: str,
                     include_images: bool = False):
        """
        Save examples to JSON file.
        
        Args:
            examples: List of examples (StrokeExample, EditExample, or TrainingExample)
            filepath: Output file path
            include_images: Include image data (large!)
        """
        data = []
        for ex in examples:
            d = ex.to_dict()
            if include_images:
                if hasattr(ex, 'canvas'):
                    d['canvas'] = ex.canvas.tolist()
                if hasattr(ex, 'stroke_image'):
                    d['stroke_image'] = ex.stroke_image.tolist()
            data.append(d)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(examples)} examples to {filepath}")
    
    def save_training_batch(self,
                           examples: List[TrainingExample],
                           output_dir: str,
                           batch_name: str = "batch"):
        """
        Save training examples with images as separate numpy files.
        
        Args:
            examples: List of TrainingExample
            output_dir: Directory to save to
            batch_name: Prefix for files
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save images as single numpy array
        images = np.stack([ex.stroke_image for ex in examples])
        np.save(output_path / f"{batch_name}_images.npy", images)
        
        # Save metadata as JSON
        metadata = [ex.to_dict() for ex in examples]
        with open(output_path / f"{batch_name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(examples)} examples to {output_dir}:")
        print(f"  - {batch_name}_images.npy: {images.shape}")
        print(f"  - {batch_name}_metadata.json")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo the dataset generator."""
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print("=" * 60)
    print("Stroke Dataset Generator")
    print("=" * 60)
    
    gen = StrokeDatasetGenerator(seed=42)
    
    print(f"\nAvailable symbols: {len(gen.available_symbols)}")
    
    # Generate stroke progression examples
    print("\n" + "-" * 40)
    print("1. Stroke Progression Examples")
    print("-" * 40)
    
    test_symbols = ['A', 'x', '+']
    for sym in test_symbols:
        if sym in gen.available_symbols:
            examples = gen.generate_stroke_progression(sym, context="f(x) = ")
            print(f"\n  Symbol '{sym}':")
            for ex in examples:
                print(f"    Stroke {ex.stroke_idx + 1}/{ex.total_strokes}: "
                      f"output='{ex.output}', points={len(ex.stroke_points)}")
    
    # Generate edit examples with multi-symbol support
    print("\n" + "-" * 40)
    print("2. Edit Examples (with multi-symbol support)")
    print("-" * 40)
    
    if HAS_CASES:
        edit_examples = gen.generate_with_cases(count=10, multi_symbol_ratio=0.5)
        print(f"\n  Generated {len(edit_examples)} edit examples:")
        for ex in edit_examples[:5]:
            print(f"    {ex.operation}: {ex.edit_symbols} -> canvas {ex.canvas.shape}")
            print(f"      Before: '{ex.before_latex[:30]}...' -> After: '{ex.after_latex[:30]}...'")
    else:
        print("  (cases.py not available)")
    
    # Generate complete training examples
    print("\n" + "-" * 40)
    print("3. Complete Training Examples")
    print("-" * 40)
    
    training_examples = gen.generate_training_examples(
        count=50,
        wait_ratio=0.4,
        multi_symbol_ratio=0.3
    )
    
    wait_count = sum(1 for e in training_examples if e.target == '<WAIT>')
    complete_count = len(training_examples) - wait_count
    multi_count = sum(1 for e in training_examples if len(e.edit_symbols) > 1)
    
    print(f"\n  Generated {len(training_examples)} training examples:")
    print(f"    <WAIT>: {wait_count}")
    print(f"    Complete: {complete_count}")
    print(f"    Multi-symbol: {multi_count}")
    
    # Show sample examples
    print("\n  Sample examples:")
    for ex in training_examples[:3]:
        print(f"    ID: {ex.id}")
        print(f"      Context: '{ex.latex_context}'")
        print(f"      Target: '{ex.target}'")
        print(f"      Symbols: {ex.edit_symbols}")
        print(f"      Image: {ex.stroke_image.shape}")
        print()
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

