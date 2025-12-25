"""
Context Composer

Composes training examples by combining chunks from a pool.
Implements various sampling strategies and curriculum learning.
"""

import random
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .chunk_types import ExpressionChunk, ChunkPool
from .training_example import ComposedTrainingExample


class ContextComposer:
    """
    Composes training examples by combining chunks from a pool.
    
    Strategies:
    1. Random context: Pick N random chunks as context
    2. Depth-matched: Pick context chunks near target depth
    3. Difficulty-matched: Match context to target difficulty
    4. Diverse: Ensure context has variety of depths/categories
    """
    
    def __init__(self, pool: ChunkPool, seed: Optional[int] = None):
        self.pool = pool
        if seed is not None:
            random.seed(seed)
    
    def compose_single(self,
                       target_chunk: ExpressionChunk,
                       num_context_chunks: int = 3,
                       context_strategy: str = "random",
                       separator: str = " \\quad ",
                       target_position: Optional[int] = None) -> ComposedTrainingExample:
        """
        Compose a single training example.
        
        Args:
            target_chunk: The chunk to edit
            num_context_chunks: Number of context chunks to add
            context_strategy: How to select context ("random", "depth_matched", "diverse")
            separator: LaTeX to join chunks
            target_position: Where to place target (None = random)
            
        Returns:
            ComposedTrainingExample
        """
        # Select context chunks based on strategy
        exclude = {target_chunk.id}
        
        if context_strategy == "random":
            context_chunks = self.pool.sample_random(num_context_chunks, exclude)
        
        elif context_strategy == "depth_matched":
            # Sample chunks within ±2 of target depth
            context_chunks = self.pool.sample_by_depth_range(
                max(1, target_chunk.depth - 2),
                target_chunk.depth + 2,
                num_context_chunks
            )
        
        elif context_strategy == "diverse":
            # Try to get chunks from different categories and depths
            context_chunks = []
            used_categories = set()
            used_depths = set()
            
            for _ in range(num_context_chunks * 3):  # Try multiple times
                candidates = self.pool.sample_random(1, exclude | {c.id for c in context_chunks})
                if not candidates:
                    break
                c = candidates[0]
                # Prefer diversity
                if c.category not in used_categories or c.depth not in used_depths:
                    context_chunks.append(c)
                    used_categories.add(c.category)
                    used_depths.add(c.depth)
                if len(context_chunks) >= num_context_chunks:
                    break
            
            # Fill remaining slots randomly
            if len(context_chunks) < num_context_chunks:
                more = self.pool.sample_random(
                    num_context_chunks - len(context_chunks),
                    exclude | {c.id for c in context_chunks}
                )
                context_chunks.extend(more)
        
        else:
            raise ValueError(f"Unknown context strategy: {context_strategy}")
        
        # Determine target position
        total_chunks = len(context_chunks) + 1
        if target_position is None:
            target_position = random.randint(0, total_chunks - 1)
        else:
            target_position = max(0, min(target_position, total_chunks - 1))
        
        # Generate edit for target chunk
        if not target_chunk.editable_positions:
            # No editable positions - skip this chunk
            raise ValueError(f"Target chunk {target_chunk.id} has no editable positions")
        
        before_latex, after_latex, edit_symbol, edit_pos_in_chunk = target_chunk.random_edit()
        
        # Find the old symbol being replaced
        old_symbol = None
        for pos, sym in target_chunk.editable_positions:
            if pos == edit_pos_in_chunk:
                old_symbol = sym
                break
        if old_symbol is None:
            old_symbol = before_latex[edit_pos_in_chunk:edit_pos_in_chunk+1]
        
        # Build full context
        all_chunks = (
            context_chunks[:target_position] + 
            [target_chunk] + 
            context_chunks[target_position:]
        )[:total_chunks]  # Ensure we don't exceed
        
        # Join chunks for before/after
        chunk_latexes_before = [
            before_latex if c.id == target_chunk.id else c.latex
            for c in all_chunks
        ]
        chunk_latexes_after = [
            after_latex if c.id == target_chunk.id else c.latex
            for c in all_chunks
        ]
        
        full_before = separator.join(chunk_latexes_before)
        full_after = separator.join(chunk_latexes_after)
        
        # Calculate EXACT position in full_context_before for minimal edit
        # Position = sum of (chunk lengths + separator lengths) before target + position within target
        chars_before_target = 0
        for i, chunk in enumerate(all_chunks):
            if i == target_position:
                break
            chars_before_target += len(chunk_latexes_before[i])
            chars_before_target += len(separator)
        
        # Now calculate absolute positions in full_before
        edit_start_pos = chars_before_target + edit_pos_in_chunk
        edit_end_pos = edit_start_pos + len(old_symbol)
        
        # The new content is just the replacement symbol - MINIMAL!
        edit_new_content = edit_symbol
        edit_old_content = old_symbol
        
        # Verify: full_before[start:end] should equal old_symbol
        actual_old = full_before[edit_start_pos:edit_end_pos]
        if actual_old != old_symbol:
            # Position calculation might be off - try to find it
            idx = full_before.find(old_symbol)
            if idx != -1:
                edit_start_pos = idx
                edit_end_pos = idx + len(old_symbol)
        
        # ESTIMATE 2D POSITION for 2D RoPE
        # x: proportional to character position in full context
        # y: based on LaTeX structure (subscripts lower, superscripts higher)
        edit_2d_x = edit_start_pos / max(1, len(full_before))
        
        # Estimate y based on LaTeX structure around edit position
        context_before_edit = full_before[:edit_start_pos]
        
        # Count open/close braces to determine nesting
        open_braces = context_before_edit.count('{')
        close_braces = context_before_edit.count('}')
        
        # Check if we're in a subscript (lower), superscript (higher), or baseline
        in_subscript = context_before_edit.count('_{') > close_braces
        in_superscript = context_before_edit.count('^{') > close_braces
        
        # Check if we're in a fraction - need to track \frac{...}{...} structure
        frac_count = context_before_edit.count(r'\frac{')
        frac_divider_count = context_before_edit.count('}{')  # divider between num/denom
        in_fraction_num = frac_count > 0 and frac_count > frac_divider_count
        in_fraction_den = frac_count > 0 and frac_divider_count >= frac_count
        
        if in_subscript:
            edit_2d_y = 0.7  # Lower
        elif in_superscript:
            edit_2d_y = 0.3  # Higher  
        elif in_fraction_num:
            edit_2d_y = 0.3  # Numerator is higher
        elif in_fraction_den:
            edit_2d_y = 0.7  # Denominator is lower
        else:
            edit_2d_y = 0.5  # Baseline
        
        edit_2d_position = (edit_2d_x, edit_2d_y)
        
        # Compute effective depth (max of all chunks + 1 for composition)
        all_depths = [c.depth for c in context_chunks] + [target_chunk.depth]
        total_depth = max(all_depths) + 1  # +1 for the composition itself
        
        return ComposedTrainingExample(
            id=f"composed_{target_chunk.id}_{random.randint(1000, 9999)}",
            context_chunks=context_chunks,
            target_chunk=target_chunk,
            target_position_in_context=target_position,
            full_context_before=full_before,
            full_context_after=full_after,
            target_before=before_latex,
            target_after=after_latex,
            # MINIMAL EDIT INFO
            edit_start_pos=edit_start_pos,
            edit_end_pos=edit_end_pos,
            edit_old_content=edit_old_content,
            edit_new_content=edit_new_content,
            # 2D POSITION for 2D RoPE attention
            edit_2d_position=edit_2d_position,
            edit_2d_bbox=None,  # Set when we have exact rendering info
            # Legacy
            edit_position=edit_pos_in_chunk,
            edit_old_symbol=old_symbol,
            edit_new_symbol=edit_symbol,
            context_depths=[c.depth for c in context_chunks],
            target_depth=target_chunk.depth,
            total_depth=total_depth,
            separator=separator,
        )
    
    def compose_batch(self,
                      num_examples: int,
                      target_depth: Optional[int] = None,
                      min_depth: Optional[int] = None,
                      max_depth: Optional[int] = None,
                      num_context_chunks: int = 3,
                      context_strategy: str = "diverse",
                      separator: str = " \\quad ",
                      max_retries: int = 3) -> List[ComposedTrainingExample]:
        """
        Compose a batch of training examples.
        
        Args:
            num_examples: Number of examples to generate
            target_depth: If specified, only use targets of exact depth
            min_depth: If specified, minimum target depth
            max_depth: If specified, maximum target depth
            num_context_chunks: Context chunks per example
            context_strategy: How to select context
            separator: LaTeX separator
            max_retries: Retries per example if target has no editable positions
            
        Returns:
            List of ComposedTrainingExample
        """
        examples = []
        used_target_ids = set()  # Avoid duplicate targets
        attempts = 0
        max_attempts = num_examples * 5  # Safety limit
        
        while len(examples) < num_examples and attempts < max_attempts:
            attempts += 1
            
            # Select target chunk (with retry for editable ones)
            target = None
            for _ in range(max_retries):
                if target_depth is not None:
                    targets = self.pool.sample_by_depth(target_depth, 5)
                elif min_depth is not None or max_depth is not None:
                    min_d = min_depth if min_depth is not None else 1
                    max_d = max_depth if max_depth is not None else 10
                    targets = self.pool.sample_by_depth_range(min_d, max_d, 5)
                else:
                    targets = self.pool.sample_random(5)
                
                # Find one with editable positions that we haven't used
                for t in targets:
                    if t.editable_positions and t.id not in used_target_ids:
                        target = t
                        break
                
                if target:
                    break
            
            if not target:
                continue
            
            try:
                example = self.compose_single(
                    target_chunk=target,
                    num_context_chunks=num_context_chunks,
                    context_strategy=context_strategy,
                    separator=separator,
                )
                examples.append(example)
                used_target_ids.add(target.id)
            except Exception as e:
                # Silently retry
                continue
        
        return examples
    
    def compose_curriculum(self,
                           total_examples: int,
                           min_context: int = 1,
                           max_context: int = 5,
                           depth_distribution: Optional[Dict[Tuple[int, int], float]] = None,
                           separator: str = " \\quad ") -> List[ComposedTrainingExample]:
        """
        Generate curriculum-ordered examples with increasing complexity by depth.
        
        Progression:
        1. Shallow depths (1-2), few context chunks
        2. Medium depths (3-4), moderate context
        3. Deep (5+), many context chunks
        
        Args:
            total_examples: Total number of examples
            min_context: Minimum context chunks
            max_context: Maximum context chunks
            depth_distribution: Optional {(min_depth, max_depth): percentage}
            separator: LaTeX separator
            
        Returns:
            List ordered by increasing depth
        """
        examples = []
        
        if depth_distribution is None:
            # Default: 40% shallow, 40% medium, 20% deep
            depth_distribution = {
                (1, 2): 0.4,   # Shallow
                (3, 4): 0.4,   # Medium
                (5, 10): 0.2,  # Deep
            }
        
        for (min_d, max_d), pct in sorted(depth_distribution.items()):
            count = int(total_examples * pct)
            # Scale context with depth
            ctx_chunks = min_context + int((max_context - min_context) * (min_d / 5))
            
            examples.extend(self.compose_batch(
                num_examples=count,
                min_depth=min_d,
                max_depth=max_d,
                num_context_chunks=min(ctx_chunks, max_context),
                context_strategy="diverse" if min_d >= 3 else "random",
                separator=separator,
            ))
        
        return examples
    
    def save_examples(self, examples: List[ComposedTrainingExample], filepath: str):
        """Save composed examples to JSON."""
        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_examples": len(examples),
                "pool_stats": self.pool.get_stats(),
            },
            "examples": [e.to_dict() for e in examples],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(examples)} composed examples to {filepath}")


# =============================================================================
# Convenience Functions
# =============================================================================

def build_training_dataset(
    seed: int = 42,
    chunks_per_depth: int = 100,
    num_training_examples: int = 5000,
    context_chunks_range: Tuple[int, int] = (1, 5),
    output_pool_path: Optional[str] = None,
    output_examples_path: Optional[str] = None,
    render_images: bool = False,
    images_dir: Optional[str] = None,
    create_edit_crops: bool = False,
) -> Tuple[ChunkPool, List[ComposedTrainingExample]]:
    """
    Build a complete training dataset using compositional augmentation.
    
    EFFICIENT RENDERING:
    - Atomic chunks are rendered ONCE to images_dir
    - Composition uses pre-rendered images (no re-rendering!)
    - Full context is NEVER rendered as one image
    
    EDIT CROP STRATEGY:
    - If create_edit_crops=False: Use full chunk image (simpler, model learns to focus)
    - If create_edit_crops=True: Pre-crop around each editable position
    
    Args:
        seed: Random seed
        chunks_per_depth: Chunks to generate per depth level
        num_training_examples: Total training examples
        context_chunks_range: (min, max) context chunks per example
        output_pool_path: Optional path to save chunk pool
        output_examples_path: Optional path to save examples
        render_images: Whether to render chunk images
        images_dir: Directory for rendered chunk images
        create_edit_crops: Also create cropped regions for each editable position
        
    Returns:
        Tuple of (ChunkPool, List[ComposedTrainingExample])
    """
    from synthetic_case_generator import CaseGenerator
    
    random.seed(seed)
    
    # Step 1: Generate chunk pool
    print("Step 1: Generating chunk pool...")
    generator = CaseGenerator(seed=seed)
    pool = ChunkPool.from_case_generator(generator, chunks_per_depth=chunks_per_depth)
    
    # Step 2: Render chunk images (ONE-TIME operation)
    if render_images and images_dir:
        print(f"\nStep 2: Rendering chunk images to {images_dir}...")
        print("  (This is done ONCE - chunks are reused for composition)")
        if create_edit_crops:
            print("  Also creating edit crops (heuristic position estimation)...")
        render_stats = pool.render_all(images_dir, verbose=True, create_edit_crops=create_edit_crops)
        print(f"  Render stats: {render_stats}")
    
    if output_pool_path:
        pool.save(output_pool_path)
    
    # Step 3: Compose training examples (fast - no rendering!)
    print(f"\nStep 3: Composing {num_training_examples} training examples...")
    print("  (No rendering needed - uses pre-rendered chunks)")
    composer = ContextComposer(pool, seed=seed)
    
    examples = composer.compose_curriculum(
        total_examples=num_training_examples,
        min_context=context_chunks_range[0],
        max_context=context_chunks_range[1],
    )
    
    if output_examples_path:
        composer.save_examples(examples, output_examples_path)
    
    # Stats
    print(f"\nDataset Statistics:")
    print(f"  Chunk Pool: {len(pool.chunks)} chunks")
    print(f"  Rendered: {pool.get_stats()['rendered_chunks']} chunks with images")
    print(f"  Training Examples: {len(examples)}")
    print(f"  Depth distribution: {pool.get_stats()['by_depth']}")
    
    return pool, examples


def render_chunk_pool(pool_path: str, images_dir: str, dpi: int = 150) -> ChunkPool:
    """
    Load a chunk pool and render all images.
    
    Use this to pre-render a saved pool:
        pool = render_chunk_pool("chunk_pool.json", "chunk_images/")
        pool.save("chunk_pool_with_images.json")
    """
    pool = ChunkPool.load(pool_path)
    print(f"\nRendering {len(pool.chunks)} chunks...")
    stats = pool.render_all(images_dir, dpi=dpi)
    print(f"Done: {stats}")
    return pool

