"""
Composed Training Example

Data structure for training examples composed from multiple chunks.
Implements minimal edit format for efficient autoregressive learning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from .chunk_types import ExpressionChunk


@dataclass 
class ComposedTrainingExample:
    """
    A training example with composed context from multiple chunks.
    
    TRAINING FORMAT (MINIMAL EDIT):
    ===============================
    
    At INFERENCE time, the model has NO knowledge of "chunks" - it only sees:
      1. Full LaTeX context (one continuous string)
      2. Edit region image (where user is writing)
    
    Therefore, the AR output must be MINIMAL - just the new symbol(s),
    not an entire artificial "chunk"!
    
    Input to model:
      1. latex_context: Full LaTeX string before edit
         "x^{2} + \\alpha = y"
         
      2. edit_region_image: Crop around where edit is happening
      
      3. edit_start_position: Character index where edit starts
      
      4. edit_end_position: Character index where old content ends
    
    Output (autoregressive target):
      NEW SYMBOL(S) ONLY!  ← Minimal output that replaces old content
      
    Example:
      Input context:  "x^{2} + \\alpha = y"
      Edit positions: start=8, end=14 (the "\\alpha" part)
      Crop image:     [stroke of β]
      AR target:      "\\beta"  ← ONLY the replacement, not surrounding!
      
    At inference, apply edit:
      result = context[:start] + model_output + context[end:]
             = "x^{2} + " + "\\beta" + " = y"
             = "x^{2} + \\beta = y"
    
    Why minimal output?
      - Model has NO chunk boundaries at inference
      - Minimal = faster decoding (1-5 tokens typically)
      - Position tells WHERE to apply the edit
      - Works for ADD (end=start), REPLACE (end>start), INSERT
    """
    id: str
    
    # The composed parts (for reference/debugging)
    context_chunks: List[ExpressionChunk]
    target_chunk: ExpressionChunk
    target_position_in_context: int  # Index where target appears in chunk list
    
    # Full LaTeX strings
    full_context_before: str  # INPUT: Full context before edit
    full_context_after: str   # For verification (not used in training)
    
    # Target chunk info (for reference)
    target_before: str        # The chunk before edit
    target_after: str         # The chunk after edit (for reference only)
    
    # MINIMAL EDIT INFO - This is what matters for training!
    edit_start_pos: int       # Character index in full_context where edit STARTS
    edit_end_pos: int         # Character index where OLD content ENDS
    edit_old_content: str     # What's being replaced (for verification)
    edit_new_content: str     # AUTOREGRESSIVE TARGET - minimal output!
    
    # Legacy fields (for backward compat)
    edit_position: int        # Character position within target chunk
    edit_old_symbol: str
    edit_new_symbol: str
    
    # Metadata
    context_depths: List[int]
    target_depth: int
    total_depth: int  # Effective depth of the composed expression
    
    # FIELDS WITH DEFAULTS must come last
    separator: str = " \\quad "  # How chunks are joined
    
    # 2D POSITIONAL INFO - For 2D RoPE attention!
    # These are normalized (0-1) coordinates of edit region center in image space
    # The model uses this to attend to nearby context tokens via 2D RoPE
    edit_2d_position: Tuple[float, float] = (0.5, 0.5)  # (x_center, y_center)
    edit_2d_bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    
    def get_edit_image_path(self) -> Optional[str]:
        """
        Get the image path for the edit region.
        
        Priority:
        1. Pre-rendered crop for this specific edit position
        2. Full chunk image (model learns to focus)
        """
        # Try specific crop first
        if self.edit_position in self.target_chunk.edit_crops:
            return self.target_chunk.edit_crops[self.edit_position]
        # Fall back to full chunk image
        return self.target_chunk.image_path
    
    def get_training_pair(self) -> Dict[str, Any]:
        """
        Get the actual training input/output pair for MINIMAL EDIT learning.
        
        The model learns to output ONLY the new content, not entire chunks.
        Position information tells WHERE to apply the edit.
        
        KEY: edit_2d_position provides the (x, y) center for 2D RoPE!
        This allows the model to attend to nearby context tokens.
        
        Returns:
            {
                "input_context": full LaTeX before edit,
                "input_edit_2d_pos": (x, y) normalized position for 2D RoPE,
                "input_edit_image": path to edit region crop,
                "output": MINIMAL - just the new symbol(s)!
            }
            
        At inference, apply: 
            result = context[:start] + output + context[end:]
        """
        return {
            # Input
            "input_context": self.full_context_before,
            "input_edit_2d_pos": self.edit_2d_position,  # (x, y) for 2D RoPE!
            "input_edit_bbox": self.edit_2d_bbox,        # Full bbox if available
            "input_edit_image": self.get_edit_image_path(),
            # String positions (for applying edit)
            "input_edit_start": self.edit_start_pos,
            "input_edit_end": self.edit_end_pos,
            # Output - MINIMAL!
            "output": self.edit_new_content,  # Just "\beta", not "\beta + y"!
            # For verification
            "_old_content": self.edit_old_content,
            "_result_preview": (
                self.full_context_before[:self.edit_start_pos] + 
                self.edit_new_content + 
                self.full_context_before[self.edit_end_pos:]
            ),
        }
    
    def verify_edit(self) -> bool:
        """Verify that applying the edit produces the expected result."""
        result = (
            self.full_context_before[:self.edit_start_pos] + 
            self.edit_new_content + 
            self.full_context_before[self.edit_end_pos:]
        )
        return result == self.full_context_after
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            # TRAINING DATA (minimal edit format)
            "input_context": self.full_context_before,      # Model INPUT (text)
            "input_edit_2d_pos": self.edit_2d_position,     # (x, y) for 2D RoPE!
            "input_edit_bbox": self.edit_2d_bbox,           # Full bbox if available
            "input_edit_start": self.edit_start_pos,        # Where edit starts (1D)
            "input_edit_end": self.edit_end_pos,            # Where old content ends
            "input_edit_image": self.get_edit_image_path(), # Model INPUT (image)
            "output": self.edit_new_content,                # Model OUTPUT - MINIMAL!
            # For verification
            "full_context_after": self.full_context_after,
            "edit_old_content": self.edit_old_content,
            # Reference info
            "context_chunk_ids": [c.id for c in self.context_chunks],
            "target_chunk_id": self.target_chunk.id,
            "target_position_in_context": self.target_position_in_context,
            "target_before": self.target_before,
            "target_after": self.target_after,
            "edit_position": self.edit_position,
            "edit_old_symbol": self.edit_old_symbol,
            "edit_new_symbol": self.edit_new_symbol,
            "context_depths": self.context_depths,
            "target_depth": self.target_depth,
            "total_depth": self.total_depth,
            "separator": self.separator,
        }

