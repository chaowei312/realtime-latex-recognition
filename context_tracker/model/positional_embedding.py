"""
Positional Embedding & Token-to-Handwriting Mapping

PURPOSE:
    Extract and save positional information from handwritten images to LaTeX context.
    Each LaTeX token stores its original handwritten shape location so that when
    the user edits later, the model can identify and update those tokens.

WORKFLOW:
    1. User writes handwritten expression (strokes on canvas)
    2. Model recognizes strokes -> LaTeX tokens with positions
    3. Save: token + bbox + stroke_ids for each recognized symbol
    4. Later: user crosses out / edits a region
    5. Model uses saved positions to identify which tokens are affected
    6. Update LaTeX accordingly (DELETE, REPLACE, etc.)

EXAMPLE:
    User writes: "x + y" (handwritten)
    
    Recognition output:
        Token "x" -> bbox(10,20,30,40), strokes=[0,1]
        Token "+" -> bbox(40,25,55,35), strokes=[2,3]  
        Token "y" -> bbox(65,20,85,40), strokes=[4,5,6]
    
    Saved context: "x + y" with position metadata
    
    Later: User crosses out "y" (strokes in region 65-85)
    Model: Detects edit in bbox(65,85) -> affects token "y"
    Result: "x + <EMPTY>" or "x +"

POSITION STRATEGY:
    - Structural tokens { } get positions based on enclosed content's bbox edges
    - { gets LEFT edge (x_min) of enclosed content
    - } gets RIGHT edge (x_max) of enclosed content
    - Content tokens get CENTER of their bounding box
    - Each token stores: position, bbox, associated stroke IDs
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box with origin at top-left."""
    x_min: float  # left edge
    y_min: float  # top edge
    x_max: float  # right edge
    y_max: float  # bottom edge
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (x, y)."""
        return ((self.x_min + self.x_max) / 2, 
                (self.y_min + self.y_max) / 2)
    
    @property
    def left(self) -> float:
        """Left edge x-coordinate."""
        return self.x_min
    
    @property
    def right(self) -> float:
        """Right edge x-coordinate."""
        return self.x_max
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @classmethod
    def from_center_size(cls, cx: float, cy: float, w: float, h: float) -> 'BoundingBox':
        """Create bbox from center and size."""
        return cls(
            x_min=cx - w/2,
            y_min=cy - h/2,
            x_max=cx + w/2,
            y_max=cy + h/2
        )
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Compute union of two bboxes."""
        return BoundingBox(
            x_min=min(self.x_min, other.x_min),
            y_min=min(self.y_min, other.y_min),
            x_max=max(self.x_max, other.x_max),
            y_max=max(self.y_max, other.y_max)
        )


@dataclass
class TokenPosition:
    """Position embedding for a single token."""
    token: str
    token_id: int
    x: float  # x-coordinate (0-1 normalized)
    y: float  # y-coordinate (0-1 normalized)
    bbox: Optional[BoundingBox] = None  # original bbox if available
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


@dataclass
class TokenHandwritingMapping:
    """
    Complete mapping between a LaTeX token and its handwritten representation.
    
    This is saved with the LaTeX context so the model can later identify
    which tokens are affected by user edits.
    
    Attributes:
        token: The LaTeX token string (e.g., 'x', '\\alpha', '+')
        token_id: Token ID from vocabulary
        token_index: Position in token sequence (0-based)
        bbox: Bounding box in image coordinates
        stroke_ids: List of stroke indices that form this symbol
        x_norm: Normalized x position (0-1)
        y_norm: Normalized y position (0-1)
        confidence: Recognition confidence (0-1)
    """
    token: str
    token_id: int
    token_index: int
    bbox: BoundingBox
    stroke_ids: List[int]
    x_norm: float
    y_norm: float
    confidence: float = 1.0
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (in image coords) is inside this token's bbox."""
        return (self.bbox.x_min <= x <= self.bbox.x_max and
                self.bbox.y_min <= y <= self.bbox.y_max)
    
    def overlaps_region(self, region: BoundingBox, threshold: float = 0.5) -> bool:
        """Check if this token overlaps with a region (e.g., an erase stroke)."""
        # Compute intersection
        x_overlap = max(0, min(self.bbox.x_max, region.x_max) - 
                          max(self.bbox.x_min, region.x_min))
        y_overlap = max(0, min(self.bbox.y_max, region.y_max) - 
                          max(self.bbox.y_min, region.y_min))
        intersection = x_overlap * y_overlap
        
        # Compute IoU with this token's area
        token_area = self.bbox.width * self.bbox.height
        if token_area == 0:
            return False
        
        overlap_ratio = intersection / token_area
        return overlap_ratio >= threshold
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'token': self.token,
            'token_id': self.token_id,
            'token_index': self.token_index,
            'bbox': {
                'x_min': self.bbox.x_min,
                'y_min': self.bbox.y_min,
                'x_max': self.bbox.x_max,
                'y_max': self.bbox.y_max,
            },
            'stroke_ids': self.stroke_ids,
            'x_norm': self.x_norm,
            'y_norm': self.y_norm,
            'confidence': self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TokenHandwritingMapping':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            token=d['token'],
            token_id=d['token_id'],
            token_index=d['token_index'],
            bbox=BoundingBox(**d['bbox']),
            stroke_ids=d['stroke_ids'],
            x_norm=d['x_norm'],
            y_norm=d['y_norm'],
            confidence=d.get('confidence', 1.0),
        )


@dataclass  
class LaTeXContext:
    """
    Complete LaTeX context with handwriting position metadata.
    
    This is the saved state after recognition, enabling later editing.
    
    Attributes:
        latex: The complete LaTeX string
        tokens: List of token strings
        token_ids: List of token IDs
        mappings: List of TokenHandwritingMapping for each content token
        image_size: (width, height) of the original image
        timestamp: When this context was created
    """
    latex: str
    tokens: List[str]
    token_ids: List[int]
    mappings: List[TokenHandwritingMapping]
    image_size: Tuple[int, int]
    timestamp: float = 0.0
    
    def find_tokens_in_region(self, region: BoundingBox, 
                               overlap_threshold: float = 0.5) -> List[TokenHandwritingMapping]:
        """
        Find all tokens that overlap with a given region.
        
        Used when user draws an erase stroke or edits a region.
        
        Args:
            region: The region being edited (e.g., erase stroke bbox)
            overlap_threshold: Minimum overlap ratio to consider affected
            
        Returns:
            List of affected TokenHandwritingMapping
        """
        affected = []
        for mapping in self.mappings:
            if mapping.overlaps_region(region, overlap_threshold):
                affected.append(mapping)
        return affected
    
    def find_tokens_by_strokes(self, stroke_ids: List[int]) -> List[TokenHandwritingMapping]:
        """
        Find tokens associated with specific strokes.
        
        Used when detecting which strokes the user is modifying.
        
        Args:
            stroke_ids: List of stroke indices being modified
            
        Returns:
            List of TokenHandwritingMapping that include any of these strokes
        """
        stroke_set = set(stroke_ids)
        affected = []
        for mapping in self.mappings:
            if stroke_set.intersection(mapping.stroke_ids):
                affected.append(mapping)
        return affected
    
    def update_after_edit(self, 
                          affected_tokens: List[TokenHandwritingMapping],
                          operation: str,
                          new_content: Optional[str] = None,
                          preserve_region: bool = True) -> 'LaTeXContext':
        """
        Create updated context after an edit operation.
        
        IMPORTANT: When erasing content inside braces (e.g., {a+b} -> {<EMPTY>}),
        the brace positions are PRESERVED. This allows:
        - { keeps LEFT edge position of original content
        - } keeps RIGHT edge position of original content
        - <EMPTY> gets CENTER of the original region
        - Model knows WHERE user can write new content
        
        Args:
            affected_tokens: Tokens being modified
            operation: 'DELETE', 'REPLACE', 'FILL'
            new_content: New content for REPLACE/FILL operations
            preserve_region: If True, keeps brace positions for REPLACE with <EMPTY>
            
        Returns:
            New LaTeXContext with updated state
        """
        # Get indices of affected tokens
        affected_indices = {m.token_index for m in affected_tokens}
        
        # Compute the bounding region of affected content (for preserving brace positions)
        affected_region = None
        if affected_tokens and preserve_region:
            x_min = min(m.bbox.x_min for m in affected_tokens)
            y_min = min(m.bbox.y_min for m in affected_tokens)
            x_max = max(m.bbox.x_max for m in affected_tokens)
            y_max = max(m.bbox.y_max for m in affected_tokens)
            affected_region = BoundingBox(x_min, y_min, x_max, y_max)
        
        new_tokens = []
        new_ids = []
        new_mappings = []
        empty_added = False  # Track if we've already added <EMPTY>
        
        for i, (token, token_id) in enumerate(zip(self.tokens, self.token_ids)):
            if i in affected_indices:
                if operation == 'DELETE':
                    # Skip this token (delete it)
                    continue
                elif operation == 'REPLACE' and new_content:
                    # Replace ALL affected tokens with ONE <EMPTY>
                    if not empty_added:
                        new_tokens.append(new_content)
                        new_ids.append(-1)  # Will be re-tokenized
                        
                        # If replacing with <EMPTY>, create a mapping that preserves region
                        if new_content == '<EMPTY>' and affected_region:
                            empty_mapping = TokenHandwritingMapping(
                                token='<EMPTY>',
                                token_id=-1,
                                token_index=len(new_tokens) - 1,
                                bbox=affected_region,  # Preserve the erased region!
                                stroke_ids=[],  # No strokes (erased)
                                x_norm=affected_region.center[0] / self.image_size[0],
                                y_norm=affected_region.center[1] / self.image_size[1],
                                confidence=1.0
                            )
                            new_mappings.append(empty_mapping)
                        empty_added = True
                    # Skip additional affected tokens (already replaced)
                        
                elif operation == 'FILL':
                    # This was <EMPTY>, fill it
                    new_tokens.append(new_content if new_content else token)
                    new_ids.append(-1)
            else:
                new_tokens.append(token)
                new_ids.append(token_id)
                # Keep mappings for unaffected tokens
                for m in self.mappings:
                    if m.token_index == i:
                        # Update token_index to new position
                        updated_m = TokenHandwritingMapping(
                            token=m.token,
                            token_id=m.token_id,
                            token_index=len(new_tokens) - 1,
                            bbox=m.bbox,
                            stroke_ids=m.stroke_ids,
                            x_norm=m.x_norm,
                            y_norm=m.y_norm,
                            confidence=m.confidence
                        )
                        new_mappings.append(updated_m)
        
        # Rebuild LaTeX string
        new_latex = ' '.join(new_tokens)  # Simplified; real impl would be smarter
        
        return LaTeXContext(
            latex=new_latex,
            tokens=new_tokens,
            token_ids=new_ids,
            mappings=new_mappings,
            image_size=self.image_size,
            timestamp=self.timestamp,
        )
    
    def get_empty_region(self) -> Optional[BoundingBox]:
        """
        Get the bounding box of the <EMPTY> token if present.
        
        This tells the model WHERE the user can write new content.
        The region preserves the original size from erased content.
        
        Returns:
            BoundingBox of empty region, or None if no <EMPTY>
        """
        for m in self.mappings:
            if m.token == '<EMPTY>':
                return m.bbox
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'latex': self.latex,
            'tokens': self.tokens,
            'token_ids': self.token_ids,
            'mappings': [m.to_dict() for m in self.mappings],
            'image_size': list(self.image_size),
            'timestamp': self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LaTeXContext':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            latex=d['latex'],
            tokens=d['tokens'],
            token_ids=d['token_ids'],
            mappings=[TokenHandwritingMapping.from_dict(m) for m in d['mappings']],
            image_size=tuple(d['image_size']),
            timestamp=d.get('timestamp', 0.0),
        )


class PositionalEmbeddingComputer:
    """
    Compute positional embeddings for LaTeX tokens based on visual layout.
    
    Strategy:
        - Opening brace { -> LEFT edge of enclosed content
        - Closing brace } -> RIGHT edge of enclosed content  
        - Content tokens -> CENTER of their bounding box
        - <EMPTY> tokens -> CENTER of the empty region
        
    This lets the model learn:
        "{ ... }" means content between left_x and right_x
    """
    
    # Structural tokens that wrap content
    OPENING_TOKENS = {'{', '\\frac{', '^{', '_{', '\\sqrt{', '\\left(', 
                      '\\left[', '\\left\\{', '\\int_{', '\\sum_{', '\\prod_{'}
    CLOSING_TOKENS = {'}', '\\right)', '\\right]', '\\right\\}'}
    
    def __init__(self, image_width: int = 128, image_height: int = 128):
        """
        Args:
            image_width: Width of rendered image for normalization
            image_height: Height of rendered image for normalization
        """
        self.image_width = image_width
        self.image_height = image_height
    
    def compute_positions(self, 
                         tokens: List[str],
                         token_ids: List[int],
                         bboxes: List[Optional[BoundingBox]]) -> List[TokenPosition]:
        """
        Compute positional embeddings for a sequence of tokens.
        
        Args:
            tokens: List of token strings
            token_ids: List of token IDs
            bboxes: List of bounding boxes (None for structural tokens)
            
        Returns:
            List of TokenPosition with computed x, y coordinates
        """
        positions = []
        
        # Stack to track opening braces and their content
        brace_stack: List[Tuple[int, str]] = []  # (index, token)
        content_ranges: Dict[int, List[int]] = {}  # opening_idx -> [content indices]
        
        # First pass: identify brace pairs and their content
        for i, token in enumerate(tokens):
            if self._is_opening(token):
                brace_stack.append((i, token))
                content_ranges[i] = []
            elif self._is_closing(token) and brace_stack:
                open_idx, open_token = brace_stack.pop()
                # This closing brace matches that opening brace
                # Content is everything between them
                content_ranges[open_idx].append(i)  # closing index
            elif brace_stack:
                # This is content inside the current brace pair
                open_idx = brace_stack[-1][0]
                content_ranges[open_idx].append(i)
        
        # Second pass: compute positions
        for i, (token, token_id, bbox) in enumerate(zip(tokens, token_ids, bboxes)):
            if self._is_opening(token):
                # Opening brace: use LEFT edge of enclosed content
                content_indices = content_ranges.get(i, [])
                content_bbox = self._compute_content_bbox(content_indices, bboxes)
                
                if content_bbox:
                    x = content_bbox.left / self.image_width
                    y = content_bbox.center[1] / self.image_height
                else:
                    # No content (e.g., <EMPTY>), use own bbox or default
                    x = 0.5
                    y = 0.5
                    
                positions.append(TokenPosition(token, token_id, x, y, bbox))
                
            elif self._is_closing(token):
                # Closing brace: use RIGHT edge of enclosed content
                # Find matching opening brace
                content_bbox = self._find_preceding_content_bbox(i, tokens, bboxes)
                
                if content_bbox:
                    x = content_bbox.right / self.image_width
                    y = content_bbox.center[1] / self.image_height
                else:
                    x = 0.5
                    y = 0.5
                    
                positions.append(TokenPosition(token, token_id, x, y, bbox))
                
            elif token == '<EMPTY>':
                # Empty slot: use center of the empty region
                # This is typically between the opening and closing braces
                x, y = self._estimate_empty_position(i, tokens, bboxes)
                positions.append(TokenPosition(token, token_id, x, y, None))
                
            else:
                # Content token: use CENTER of its bounding box
                if bbox:
                    cx, cy = bbox.center
                    x = cx / self.image_width
                    y = cy / self.image_height
                else:
                    # Fallback: estimate from neighbors
                    x, y = self._estimate_from_neighbors(i, positions)
                    
                positions.append(TokenPosition(token, token_id, x, y, bbox))
        
        return positions
    
    def _is_opening(self, token: str) -> bool:
        """Check if token is an opening structural token."""
        return token in self.OPENING_TOKENS or token == '{'
    
    def _is_closing(self, token: str) -> bool:
        """Check if token is a closing structural token."""
        return token in self.CLOSING_TOKENS or token == '}'
    
    def _compute_content_bbox(self, 
                              indices: List[int], 
                              bboxes: List[Optional[BoundingBox]]) -> Optional[BoundingBox]:
        """Compute union bbox of content at given indices."""
        content_bboxes = [bboxes[i] for i in indices if i < len(bboxes) and bboxes[i]]
        
        if not content_bboxes:
            return None
        
        result = content_bboxes[0]
        for bbox in content_bboxes[1:]:
            result = result.union(bbox)
        
        return result
    
    def _find_preceding_content_bbox(self,
                                      closing_idx: int,
                                      tokens: List[str],
                                      bboxes: List[Optional[BoundingBox]]) -> Optional[BoundingBox]:
        """Find content bbox for the closing brace by looking backwards."""
        # Walk backwards to find matching opening brace
        depth = 0
        content_indices = []
        
        for i in range(closing_idx - 1, -1, -1):
            token = tokens[i]
            if self._is_closing(token):
                depth += 1
            elif self._is_opening(token):
                if depth == 0:
                    # Found matching opening brace
                    break
                depth -= 1
            else:
                if depth == 0:
                    content_indices.append(i)
        
        return self._compute_content_bbox(content_indices, bboxes)
    
    def _estimate_empty_position(self,
                                  empty_idx: int,
                                  tokens: List[str],
                                  bboxes: List[Optional[BoundingBox]]) -> Tuple[float, float]:
        """Estimate position for <EMPTY> token based on surrounding structure."""
        # Look for surrounding content/braces to estimate position
        left_x = 0.0
        right_x = 1.0
        y = 0.5
        
        # Look left for opening brace or content
        for i in range(empty_idx - 1, -1, -1):
            if bboxes[i]:
                left_x = bboxes[i].right / self.image_width
                y = bboxes[i].center[1] / self.image_height
                break
        
        # Look right for closing brace or content
        for i in range(empty_idx + 1, len(tokens)):
            if bboxes[i]:
                right_x = bboxes[i].left / self.image_width
                y = bboxes[i].center[1] / self.image_height
                break
        
        # <EMPTY> position is center of the empty region
        x = (left_x + right_x) / 2
        
        return (x, y)
    
    def _estimate_from_neighbors(self,
                                  idx: int,
                                  positions: List[TokenPosition]) -> Tuple[float, float]:
        """Estimate position from already-computed neighbor positions."""
        if not positions:
            return (0.5, 0.5)
        
        # Use last computed position as estimate
        last = positions[-1]
        return (last.x + 0.1, last.y)  # Slight offset


def compute_token_positions_from_latex_render(
    tokens: List[str],
    token_ids: List[int],
    char_bboxes: Dict[int, BoundingBox],  # character index -> bbox
    image_size: Tuple[int, int] = (128, 128)
) -> List[TokenPosition]:
    """
    Convenience function to compute positions from rendered LaTeX.
    
    Args:
        tokens: Tokenized LaTeX
        token_ids: Token IDs
        char_bboxes: Bounding boxes from PDF/image extraction
        image_size: (width, height) of rendered image
        
    Returns:
        List of TokenPosition
    """
    computer = PositionalEmbeddingComputer(image_size[0], image_size[1])
    
    # Map tokens to bboxes (structural tokens get None)
    bboxes = []
    char_idx = 0
    
    for token in tokens:
        if computer._is_opening(token) or computer._is_closing(token):
            bboxes.append(None)
        elif token == '<EMPTY>':
            bboxes.append(None)
        elif token.startswith('<'):
            # Other special tokens
            bboxes.append(None)
        else:
            # Content token - try to get its bbox
            bbox = char_bboxes.get(char_idx)
            bboxes.append(bbox)
            char_idx += len(token)  # Advance by token length
    
    return computer.compute_positions(tokens, token_ids, bboxes)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("Token-to-Handwriting Mapping: Full Workflow Example")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: User writes "x + y" (handwritten)
    # =========================================================================
    print("\n1. USER WRITES 'x + y' (handwritten strokes)")
    print("-" * 50)
    
    # After recognition, we have:
    x_bbox = BoundingBox(x_min=10, y_min=20, x_max=30, y_max=40)
    plus_bbox = BoundingBox(x_min=40, y_min=25, x_max=55, y_max=35)
    y_bbox = BoundingBox(x_min=65, y_min=20, x_max=85, y_max=40)
    
    # Create mappings for each recognized symbol
    mappings = [
        TokenHandwritingMapping(
            token='x', token_id=23, token_index=0,
            bbox=x_bbox, stroke_ids=[0, 1],
            x_norm=0.156, y_norm=0.234
        ),
        TokenHandwritingMapping(
            token='+', token_id=50, token_index=1,
            bbox=plus_bbox, stroke_ids=[2, 3],
            x_norm=0.371, y_norm=0.234
        ),
        TokenHandwritingMapping(
            token='y', token_id=24, token_index=2,
            bbox=y_bbox, stroke_ids=[4, 5, 6],
            x_norm=0.586, y_norm=0.234
        ),
    ]
    
    # Save as LaTeXContext
    context = LaTeXContext(
        latex="x + y",
        tokens=['x', '+', 'y'],
        token_ids=[23, 50, 24],
        mappings=mappings,
        image_size=(128, 128),
        timestamp=1234567890.0
    )
    
    print(f"  Recognized LaTeX: {context.latex}")
    print(f"  Saved mappings:")
    for m in context.mappings:
        print(f"    '{m.token}' at bbox({m.bbox.x_min:.0f},{m.bbox.y_min:.0f},"
              f"{m.bbox.x_max:.0f},{m.bbox.y_max:.0f}), strokes={m.stroke_ids}")
    
    # =========================================================================
    # Step 2: Later - User crosses out "y" (erase stroke in region 60-90)
    # =========================================================================
    print("\n2. USER CROSSES OUT 'y' (erase stroke in region 60-90)")
    print("-" * 50)
    
    # Erase stroke creates a region
    erase_region = BoundingBox(x_min=60, y_min=15, x_max=90, y_max=45)
    
    # Find affected tokens
    affected = context.find_tokens_in_region(erase_region, overlap_threshold=0.5)
    
    print(f"  Erase region: ({erase_region.x_min}, {erase_region.y_min}) to "
          f"({erase_region.x_max}, {erase_region.y_max})")
    print(f"  Affected tokens: {[m.token for m in affected]}")
    
    # =========================================================================
    # Step 3: Update context - DELETE the affected token
    # =========================================================================
    print("\n3. UPDATE CONTEXT (DELETE 'y')")
    print("-" * 50)
    
    new_context = context.update_after_edit(affected, operation='DELETE')
    
    print(f"  New LaTeX: {new_context.latex}")
    print(f"  New tokens: {new_context.tokens}")
    
    # =========================================================================
    # Step 4: Or - REPLACE with <EMPTY> for FILL later
    # =========================================================================
    print("\n4. ALTERNATIVE: REPLACE 'y' with <EMPTY>")
    print("-" * 50)
    
    new_context_empty = context.update_after_edit(
        affected, operation='REPLACE', new_content='<EMPTY>'
    )
    
    print(f"  New LaTeX: {new_context_empty.latex}")
    print(f"  User can now FILL the <EMPTY> with new content")
    
    # =========================================================================
    # Step 5: Serialize for storage
    # =========================================================================
    print("\n5. SERIALIZE FOR STORAGE")
    print("-" * 50)
    
    context_json = json.dumps(context.to_dict(), indent=2)
    print(f"  JSON (first 300 chars):")
    print(f"  {context_json[:300]}...")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Token-Handwriting Mapping Workflow")
    print("=" * 70)
    print("""
    1. RECOGNITION: strokes -> tokens with bbox + stroke_ids
    2. SAVE: LaTeXContext with all mappings
    3. EDIT: User draws erase/edit stroke
    4. LOOKUP: Find affected tokens by region overlap
    5. UPDATE: DELETE / REPLACE / FILL affected tokens
    6. RESULT: New LaTeXContext for continued editing
    """)

