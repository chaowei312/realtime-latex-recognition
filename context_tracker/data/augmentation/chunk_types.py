"""
Expression Chunk Types

Core data structures for compositional data augmentation:
    - ExpressionChunk: Atomic LaTeX expression unit
    - ChunkPool: Collection of chunks organized by depth/difficulty
"""

import os
import re
import random
import json
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .symbol_extraction import get_all_symbol_bboxes, extract_symbol_positions_from_pdf


# =============================================================================
# Expression Chunk - Atomic unit for composition
# =============================================================================

@dataclass
class ExpressionChunk:
    """
    An atomic LaTeX expression chunk that can be composed into longer contexts.
    
    Chunks are the building blocks for compositional data augmentation.
    Each chunk has:
    - latex: The LaTeX expression
    - depth: Semantic tree depth (from complexity analysis)
    - difficulty: easy/medium/hard classification
    - category: Type of expression (algebra, calculus, diagram, etc.)
    - editable_positions: List of (position, symbol) that can be edited
    - image_path: Pre-rendered image (render ONCE, reuse forever)
    
    RENDERING STRATEGY:
    - Each chunk is rendered to image ONCE during pool creation
    - During composition, we use pre-rendered images (no re-rendering!)
    - Full context image is NEVER rendered (wasteful)
    - Only the TARGET chunk image is used for edit crop
    """
    id: str
    latex: str
    depth: int
    difficulty: str
    difficulty_score: float
    category: str
    subcategory: str = ""
    
    # Positions where edits can be made: [(char_index, symbol_str), ...]
    editable_positions: List[Tuple[int, str]] = field(default_factory=list)
    
    # Pre-rendered image path (render once, reuse)
    image_path: Optional[str] = None
    
    # Edit region crops: {edit_position: image_path}
    # For complex chunks, we pre-render crops around each editable position
    edit_crops: Dict[int, str] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.editable_positions:
            self.editable_positions = self._find_editable_positions()
    
    def _find_editable_positions(self) -> List[Tuple[int, str]]:
        """
        Find positions in the LaTeX where symbols can be safely edited.
        
        Rules:
        - Variables: standalone letters NOT part of LaTeX commands
        - Greek letters: full commands like \\alpha
        - Digits: standalone numbers NOT in commands
        
        We exclude:
        - Letters inside \\command names
        - Letters inside environment names
        - Letters in arrow specifications like [r, d]
        """
        positions = []
        
        # First, find all LaTeX command spans to exclude
        command_spans = []
        for m in re.finditer(r'\\[a-zA-Z]+', self.latex):
            command_spans.append((m.start(), m.end()))
        
        # Also exclude content inside square brackets (arrow specs)
        for m in re.finditer(r'\[[^\]]*\]', self.latex):
            command_spans.append((m.start(), m.end()))
        
        # Also exclude content inside quotes (labels)
        for m in re.finditer(r'"[^"]*"', self.latex):
            command_spans.append((m.start(), m.end()))
        
        # Also exclude mathcal, mathbb, etc arguments
        for m in re.finditer(r'\\math[a-z]+\{[^}]*\}', self.latex):
            command_spans.append((m.start(), m.end()))
        
        def is_in_excluded(pos):
            return any(start <= pos < end for start, end in command_spans)
        
        # Find Greek letters FIRST (they are complete units)
        greek_pattern = r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega|Gamma|Delta|Theta|Lambda|Sigma|Phi|Psi|Omega)\b'
        for m in re.finditer(greek_pattern, self.latex):
            positions.append((m.start(), m.group(0)))
        
        # Find standalone uppercase letters (variables like A, B, C)
        # Must be surrounded by non-letters and not in commands
        for m in re.finditer(r'(?<![a-zA-Z\\])([A-Z])(?![a-zA-Z])', self.latex):
            if not is_in_excluded(m.start(1)):
                positions.append((m.start(1), m.group(1)))
        
        # Find standalone lowercase letters (variables like x, y, z)
        # More restrictive - must be math context
        for m in re.finditer(r'(?<![a-zA-Z\\])([a-z])(?![a-zA-Z])', self.latex):
            if not is_in_excluded(m.start(1)):
                # Extra check: must look like a math variable context
                # Skip if it looks like it's in a word
                pos = m.start(1)
                before = self.latex[max(0, pos-2):pos]
                after = self.latex[pos+1:pos+3]
                if not re.match(r'[a-z]{2}', before + after):
                    positions.append((pos, m.group(1)))
        
        # Find standalone digits (coefficients)
        for m in re.finditer(r'(?<![0-9a-zA-Z])([0-9])(?![0-9a-zA-Z])', self.latex):
            if not is_in_excluded(m.start(1)):
                positions.append((m.start(1), m.group(1)))
        
        return positions
    
    def apply_edit(self, position: int, new_symbol: str) -> str:
        """Apply an edit at the given position, returning new LaTeX."""
        # Find the editable item at this position
        for pos, old_symbol in self.editable_positions:
            if pos == position:
                return self.latex[:pos] + new_symbol + self.latex[pos + len(old_symbol):]
        raise ValueError(f"No editable position at index {position}")
    
    def random_edit(self, symbol_pool: Optional[List[str]] = None) -> Tuple[str, str, str, int]:
        """
        Generate a random edit for this chunk.
        
        Returns:
            Tuple of (before_latex, after_latex, edit_symbol, position)
        """
        if not self.editable_positions:
            raise ValueError("No editable positions in this chunk")
        
        # Default symbol pool
        if symbol_pool is None:
            symbol_pool = list("abcdefghijklmnopqrstuvwxyz") + [
                r"\alpha", r"\beta", r"\gamma", r"\delta", r"\theta",
                r"\lambda", r"\sigma", r"\phi", r"\omega"
            ]
        
        # Pick random position
        pos, old_symbol = random.choice(self.editable_positions)
        
        # Pick new symbol (MUST be different from old)
        candidates = [s for s in symbol_pool if s != old_symbol]
        if not candidates:
            raise ValueError(f"No alternative symbols for '{old_symbol}'")
        new_symbol = random.choice(candidates)
        
        after_latex = self.apply_edit(pos, new_symbol)
        
        return self.latex, after_latex, new_symbol, pos
    
    def estimate_symbol_region(self, edit_position: int, 
                                 image_width: int, image_height: int,
                                 padding_pct: float = 0.15) -> Tuple[int, int, int, int]:
        """
        Estimate pixel region for a symbol based on its LaTeX string position.
        
        Heuristic: relative position in LaTeX ≈ relative position in image
        This works reasonably for left-to-right expressions.
        
        For vertical structures (fractions, matrices), we use the full height.
        
        Args:
            edit_position: Character index in self.latex
            image_width: Width of rendered image
            image_height: Height of rendered image
            padding_pct: Extra padding around estimated region (0.15 = 15%)
            
        Returns:
            (x, y, width, height) in pixels
        """
        if not self.latex:
            return (0, 0, image_width, image_height)
        
        # Find the symbol at this position
        symbol = None
        symbol_len = 1
        for pos, sym in self.editable_positions:
            if pos == edit_position:
                symbol = sym
                symbol_len = len(sym)
                break
        
        # Estimate horizontal position (works for L-to-R math)
        # Normalize to 0-1 range
        rel_start = edit_position / len(self.latex)
        rel_end = (edit_position + symbol_len) / len(self.latex)
        rel_center = (rel_start + rel_end) / 2
        
        # Convert to pixels with padding
        region_width = max(0.2, (rel_end - rel_start) + padding_pct * 2)  # At least 20%
        x_center = rel_center * image_width
        
        x = max(0, int(x_center - region_width * image_width / 2))
        w = min(image_width - x, int(region_width * image_width))
        
        # For vertical: check if this is in a fraction/matrix structure
        has_vertical = any(cmd in self.latex for cmd in [r'\frac', r'\begin{', r'_', '^'])
        
        if has_vertical:
            # Use full height for vertical structures
            y, h = 0, image_height
        else:
            # Horizontal expression - can crop vertically too
            y = int(image_height * 0.1)
            h = int(image_height * 0.8)
        
        return (x, y, w, h)
    
    def get_edit_image_path(self, edit_position: int) -> Optional[str]:
        """
        Get the image path for a specific edit position.
        
        Returns edit_crops[position] if available, otherwise full image_path.
        """
        if edit_position in self.edit_crops:
            return self.edit_crops[edit_position]
        return self.image_path
    
    def extract_exact_symbol_positions(self) -> List[Dict]:
        """
        Extract EXACT symbol positions by rendering to PDF and using PyMuPDF.
        
        This gives precise bounding boxes for each symbol, much better than
        the heuristic estimation.
        
        Returns:
            List of {text, bbox, x0, y0, x1, y1, center, width, height}
        """
        symbols, pdf_path = get_all_symbol_bboxes(self.latex)
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        return symbols
    
    def create_accurate_crop(self, edit_position: int, 
                             padding_pct: float = 0.2,
                             output_path: Optional[str] = None) -> Optional[Any]:
        """
        Create an accurate crop around a symbol using PDF-extracted positions.
        
        This renders the chunk, extracts exact symbol bboxes, and crops
        around the target symbol with padding.
        
        Args:
            edit_position: Character index of symbol in self.latex
            padding_pct: Extra padding as fraction of bbox size
            output_path: If provided, save crop to this path
            
        Returns:
            PIL Image of crop (or None if failed)
        """
        try:
            from PIL import Image
            from generate_tikzcd import render_tikzcd_to_pdf
            import fitz
        except ImportError as e:
            print(f"Missing dependency: {e}")
            return None
        
        # Find the symbol at this position
        target_symbol = None
        for pos, sym in self.editable_positions:
            if pos == edit_position:
                # Convert LaTeX command to rendered symbol
                target_symbol = sym
                # Handle Greek letter conversion
                greek_map = {
                    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
                    r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ',
                    r'\sigma': 'σ', r'\phi': 'φ', r'\psi': 'ψ', r'\omega': 'ω',
                    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
                    r'\Sigma': 'Σ', r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
                }
                if target_symbol in greek_map:
                    target_symbol = greek_map[target_symbol]
                break
        
        if not target_symbol:
            return None
        
        # Render to PDF and extract positions
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pdf_path = f.name
        
        try:
            success = render_tikzcd_to_pdf(self.latex, pdf_path)
            if not success:
                return None
            
            # Extract symbol positions
            symbols = extract_symbol_positions_from_pdf(pdf_path)
            
            # Find target symbol (handle spaces and variations)
            target_bbox = None
            for sym in symbols:
                sym_text = sym["text"].strip()
                if sym_text == target_symbol or sym_text == target_symbol.strip():
                    target_bbox = sym
                    break
            
            # If not found, try partial match
            if not target_bbox:
                for sym in symbols:
                    if target_symbol in sym["text"] or sym["text"].strip() in target_symbol:
                        target_bbox = sym
                        break
            
            if not target_bbox:
                # Symbol not found - fall back to heuristic
                return None
            
            # Convert PDF to image and crop
            doc = fitz.open(pdf_path)
            page = doc[0]
            
            # Get page size
            page_rect = page.rect
            
            # Calculate crop region with padding
            pad_x = target_bbox["width"] * padding_pct
            pad_y = target_bbox["height"] * padding_pct
            
            crop_rect = fitz.Rect(
                max(0, target_bbox["x0"] - pad_x),
                max(0, target_bbox["y0"] - pad_y),
                min(page_rect.width, target_bbox["x1"] + pad_x),
                min(page_rect.height, target_bbox["y1"] + pad_y)
            )
            
            # Render cropped region at high resolution
            mat = fitz.Matrix(3, 3)  # 3x zoom for quality
            clip = crop_rect
            pix = page.get_pixmap(matrix=mat, clip=clip)
            
            # Convert to PIL
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            doc.close()
            
            if output_path:
                img.save(output_path)
            
            return img
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "latex": self.latex,
            "depth": self.depth,
            "difficulty": self.difficulty,
            "difficulty_score": self.difficulty_score,
            "category": self.category,
            "subcategory": self.subcategory,
            "editable_positions": self.editable_positions,
            "image_path": self.image_path,
            "edit_crops": self.edit_crops,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExpressionChunk":
        return cls(
            id=data["id"],
            latex=data["latex"],
            depth=data["depth"],
            difficulty=data["difficulty"],
            difficulty_score=data["difficulty_score"],
            category=data["category"],
            subcategory=data.get("subcategory", ""),
            editable_positions=[tuple(p) for p in data.get("editable_positions", [])],
            image_path=data.get("image_path"),
            edit_crops={int(k): v for k, v in data.get("edit_crops", {}).items()},
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Chunk Pool - Collection of chunks organized by depth
# =============================================================================

@dataclass
class ChunkPool:
    """
    A pool of expression chunks organized by depth for efficient sampling.
    
    The pool enables:
    - Sampling chunks by specific depth
    - Sampling chunks by depth range
    - Balanced sampling across difficulties
    - Persistence (save/load)
    """
    chunks: List[ExpressionChunk] = field(default_factory=list)
    
    # Indices for fast lookup
    _by_depth: Dict[int, List[int]] = field(default_factory=dict, repr=False)
    _by_difficulty: Dict[str, List[int]] = field(default_factory=dict, repr=False)
    _by_category: Dict[str, List[int]] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        self._rebuild_indices()
    
    def _rebuild_indices(self):
        """Rebuild lookup indices."""
        self._by_depth = {}
        self._by_difficulty = {}
        self._by_category = {}
        
        for i, chunk in enumerate(self.chunks):
            # By depth
            if chunk.depth not in self._by_depth:
                self._by_depth[chunk.depth] = []
            self._by_depth[chunk.depth].append(i)
            
            # By difficulty
            if chunk.difficulty not in self._by_difficulty:
                self._by_difficulty[chunk.difficulty] = []
            self._by_difficulty[chunk.difficulty].append(i)
            
            # By category
            if chunk.category not in self._by_category:
                self._by_category[chunk.category] = []
            self._by_category[chunk.category].append(i)
    
    def add(self, chunk: ExpressionChunk):
        """Add a chunk to the pool."""
        idx = len(self.chunks)
        self.chunks.append(chunk)
        
        # Update indices
        if chunk.depth not in self._by_depth:
            self._by_depth[chunk.depth] = []
        self._by_depth[chunk.depth].append(idx)
        
        if chunk.difficulty not in self._by_difficulty:
            self._by_difficulty[chunk.difficulty] = []
        self._by_difficulty[chunk.difficulty].append(idx)
        
        if chunk.category not in self._by_category:
            self._by_category[chunk.category] = []
        self._by_category[chunk.category].append(idx)
    
    def sample_by_depth(self, depth: int, count: int = 1) -> List[ExpressionChunk]:
        """Sample chunks of a specific depth."""
        if depth not in self._by_depth:
            return []
        indices = self._by_depth[depth]
        sampled = random.sample(indices, min(count, len(indices)))
        return [self.chunks[i] for i in sampled]
    
    def sample_by_depth_range(self, min_depth: int, max_depth: int, 
                               count: int = 1) -> List[ExpressionChunk]:
        """Sample chunks within a depth range."""
        eligible = []
        for d in range(min_depth, max_depth + 1):
            if d in self._by_depth:
                eligible.extend(self._by_depth[d])
        
        if not eligible:
            return []
        sampled = random.sample(eligible, min(count, len(eligible)))
        return [self.chunks[i] for i in sampled]
    
    def sample_by_difficulty(self, difficulty: str, count: int = 1) -> List[ExpressionChunk]:
        """Sample chunks of a specific difficulty."""
        if difficulty not in self._by_difficulty:
            return []
        indices = self._by_difficulty[difficulty]
        sampled = random.sample(indices, min(count, len(indices)))
        return [self.chunks[i] for i in sampled]
    
    def sample_random(self, count: int = 1, 
                      exclude_ids: Optional[set] = None) -> List[ExpressionChunk]:
        """Sample random chunks, optionally excluding specific IDs."""
        if exclude_ids:
            eligible = [i for i, c in enumerate(self.chunks) if c.id not in exclude_ids]
        else:
            eligible = list(range(len(self.chunks)))
        
        if not eligible:
            return []
        sampled = random.sample(eligible, min(count, len(eligible)))
        return [self.chunks[i] for i in sampled]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pool."""
        rendered_count = sum(1 for c in self.chunks if c.image_path)
        return {
            "total_chunks": len(self.chunks),
            "rendered_chunks": rendered_count,
            "by_depth": {d: len(indices) for d, indices in sorted(self._by_depth.items())},
            "by_difficulty": {d: len(indices) for d, indices in self._by_difficulty.items()},
            "by_category": {c: len(indices) for c, indices in self._by_category.items()},
            "depth_range": (min(self._by_depth.keys()), max(self._by_depth.keys())) if self._by_depth else (0, 0),
        }
    
    def render_all(self, output_dir: str, dpi: int = 150, 
                   skip_existing: bool = True, verbose: bool = True,
                   create_edit_crops: bool = False) -> Dict[str, int]:
        """
        Pre-render all chunks to images (ONE-TIME operation).
        
        This is the key optimization: render each atomic chunk ONCE,
        then reuse during composition (no full-context rendering needed).
        
        Args:
            output_dir: Directory to save chunk images
            dpi: Resolution for rendering
            skip_existing: Skip chunks that already have images
            verbose: Print progress
            create_edit_crops: Also create cropped regions for each editable position
            
        Returns:
            Stats dict with success/fail counts
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if create_edit_crops:
            (output_path / "crops").mkdir(exist_ok=True)
        
        # Try to import renderer
        try:
            from generate_tikzcd import render_tikzcd_to_image
            from PIL import Image
        except ImportError as e:
            print(f"Warning: Required module not available ({e}), skipping rendering")
            return {"skipped": len(self.chunks), "rendered": 0, "failed": 0, "crops": 0}
        
        stats = {"rendered": 0, "skipped": 0, "failed": 0, "crops": 0}
        
        for i, chunk in enumerate(self.chunks):
            # Skip if already rendered
            if skip_existing and chunk.image_path and os.path.exists(chunk.image_path):
                stats["skipped"] += 1
                continue
            
            # Render full chunk
            img_filename = f"{chunk.id}.png"
            img_path = output_path / img_filename
            
            try:
                success = render_tikzcd_to_image(chunk.latex, str(img_path), dpi=dpi)
                if success:
                    chunk.image_path = str(img_path)
                    stats["rendered"] += 1
                    
                    # Optionally create edit crops
                    if create_edit_crops and chunk.editable_positions:
                        try:
                            img = Image.open(img_path)
                            w, h = img.size
                            
                            for pos, symbol in chunk.editable_positions:
                                # Estimate region for this edit position
                                x, y, cw, ch = chunk.estimate_symbol_region(pos, w, h)
                                
                                # Crop and save
                                crop_img = img.crop((x, y, x + cw, y + ch))
                                crop_filename = f"{chunk.id}_edit_{pos}.png"
                                crop_path = output_path / "crops" / crop_filename
                                crop_img.save(str(crop_path))
                                
                                chunk.edit_crops[pos] = str(crop_path)
                                stats["crops"] += 1
                            
                            img.close()
                        except Exception as e:
                            if verbose:
                                print(f"  Failed to create crops for {chunk.id}: {e}")
                else:
                    stats["failed"] += 1
            except Exception as e:
                if verbose:
                    print(f"  Failed to render {chunk.id}: {e}")
                stats["failed"] += 1
            
            # Progress
            if verbose and (i + 1) % 100 == 0:
                print(f"  Rendered {i + 1}/{len(self.chunks)} chunks...")
        
        if verbose:
            print(f"Rendering complete: {stats}")
        
        return stats
    
    def create_runtime_crop(self, chunk: ExpressionChunk, edit_position: int) -> Optional[Any]:
        """
        Create a crop at runtime (on-the-fly) for a specific edit position.
        
        Use this when you don't want to pre-render all crops.
        Requires the chunk's full image to exist.
        
        Args:
            chunk: The chunk to crop
            edit_position: Position of the symbol being edited
            
        Returns:
            PIL Image of the cropped region, or None if failed
        """
        if not chunk.image_path:
            return None
        
        try:
            from PIL import Image
            
            if not os.path.exists(chunk.image_path):
                return None
            
            img = Image.open(chunk.image_path)
            w, h = img.size
            
            x, y, cw, ch = chunk.estimate_symbol_region(edit_position, w, h)
            crop = img.crop((x, y, x + cw, y + ch))
            
            return crop
        except Exception:
            return None
    
    def get_chunks_with_images(self) -> List[ExpressionChunk]:
        """Get only chunks that have been rendered."""
        return [c for c in self.chunks if c.image_path]
    
    def save(self, filepath: str):
        """Save pool to JSON file."""
        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_chunks": len(self.chunks),
                "stats": self.get_stats(),
            },
            "chunks": [c.to_dict() for c in self.chunks],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.chunks)} chunks to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ChunkPool":
        """Load pool from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        chunks = [ExpressionChunk.from_dict(c) for c in data["chunks"]]
        pool = cls(chunks=chunks)
        print(f"Loaded {len(chunks)} chunks from {filepath}")
        return pool
    
    @classmethod
    def from_case_generator(cls, generator, chunks_per_depth: int = 50,
                            max_depth: int = 7) -> "ChunkPool":
        """
        Build a chunk pool from CaseGenerator.
        
        Args:
            generator: CaseGenerator instance
            chunks_per_depth: Target number of chunks per depth level
            max_depth: Maximum depth to generate
            
        Returns:
            ChunkPool with diverse chunks
        """
        from synthetic_case_generator import analyze_latex_complexity
        
        pool = cls()
        
        # Generate cases and convert to chunks
        all_cases = generator.generate_all(count_per_category=chunks_per_depth * 2)
        
        for case in all_cases:
            # Analyze the "after" latex (the complete expression)
            complexity = analyze_latex_complexity(case.after_latex)
            
            chunk = ExpressionChunk(
                id=f"chunk_{case.id}",
                latex=case.after_latex,
                depth=complexity.depth,
                difficulty=complexity.difficulty,
                difficulty_score=complexity.difficulty_score,
                category=case.category,
                subcategory=case.subcategory,
                metadata={
                    "original_case_id": case.id,
                    "operation": case.operation,
                }
            )
            pool.add(chunk)
        
        print(f"Built pool with {len(pool.chunks)} chunks")
        print(f"Depth distribution: {pool.get_stats()['by_depth']}")
        
        return pool

