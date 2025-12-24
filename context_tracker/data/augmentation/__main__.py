#!/usr/bin/env python
"""
CLI entry point for augmentation package.

Usage:
    python -m augmentation --help
    python -m augmentation --chunks-per-depth 20 --num-examples 50 --preview 5
"""

import argparse
from .composer import build_training_dataset


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compositional Data Augmentation for LaTeX Editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (no rendering)
  python -m augmentation --chunks-per-depth 20 --num-examples 50 --preview 5
  
  # Build pool + render images (ONE-TIME)
  python -m augmentation --chunks-per-depth 100 --render --images-dir chunk_images/
  
  # Full pipeline with outputs
  python -m augmentation \\
      --chunks-per-depth 100 \\
      --num-examples 5000 \\
      --render --images-dir chunk_images/ \\
      --output-pool chunk_pool.json \\
      --output-examples training_data.json
        """
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--chunks-per-depth", type=int, default=50, 
                        help="Chunks to generate per depth level")
    parser.add_argument("--num-examples", "-n", type=int, default=1000,
                        help="Number of training examples to generate")
    parser.add_argument("--min-context", type=int, default=1,
                        help="Minimum context chunks per example")
    parser.add_argument("--max-context", type=int, default=5,
                        help="Maximum context chunks per example")
    parser.add_argument("--render", action="store_true",
                        help="Render chunk images (one-time operation)")
    parser.add_argument("--create-crops", action="store_true",
                        help="Also create cropped regions around each editable symbol")
    parser.add_argument("--images-dir", type=str, default="chunk_images",
                        help="Directory for rendered chunk images")
    parser.add_argument("--output-pool", type=str, default=None,
                        help="Output path for chunk pool JSON")
    parser.add_argument("--output-examples", type=str, default=None,
                        help="Output path for composed examples JSON")
    parser.add_argument("--preview", "-p", type=int, default=3,
                        help="Number of examples to preview")
    
    args = parser.parse_args()
    
    # Build dataset
    pool, examples = build_training_dataset(
        seed=args.seed,
        chunks_per_depth=args.chunks_per_depth,
        num_training_examples=args.num_examples,
        context_chunks_range=(args.min_context, args.max_context),
        output_pool_path=args.output_pool,
        output_examples_path=args.output_examples,
        render_images=args.render,
        images_dir=args.images_dir if args.render else None,
        create_edit_crops=args.create_crops,
    )
    
    # Preview
    if args.preview > 0:
        print(f"\n{'='*60}")
        print(f"Sample Composed Examples:")
        print(f"{'='*60}")
        
        for ex in examples[:args.preview]:
            training = ex.get_training_pair()
            print(f"\n[{ex.id}]")
            print(f"  Target depth: {ex.target_depth}, Context depths: {ex.context_depths}")
            print(f"  Edit: '{ex.edit_old_symbol}' -> '{ex.edit_new_symbol}'")
            print(f"  INPUT context: \"{training['input_context'][:60]}...\"")
            print(f"  OUTPUT target: \"{training['output']}\"")
            if ex.target_chunk.image_path:
                print(f"  Image: {ex.target_chunk.image_path}")


if __name__ == "__main__":
    main()

