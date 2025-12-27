"""
Data path configuration for Context Tracker.

Update these paths to match your local setup.
"""

from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# MathWriting dataset path
# Option 1: Inside project (default)
MATHWRITING_PATH = PROJECT_ROOT / "context_tracker" / "data" / "mathwriting"

# Option 2: External path (uncomment and modify if data is elsewhere)
# MATHWRITING_PATH = Path("D:/Datasets/mathwriting")
# MATHWRITING_PATH = Path("/data/mathwriting")

# Stroke corpus for artifacts
STROKE_CORPUS_PATH = PROJECT_ROOT / "context_tracker" / "data" / "stroke_corpus"

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "context_tracker" / "checkpoints"

# Verify paths exist
def check_data_paths():
    """Check if data paths exist and print status."""
    print("Data Path Configuration:")
    print("=" * 50)
    
    paths = {
        "MathWriting": MATHWRITING_PATH,
        "Stroke Corpus": STROKE_CORPUS_PATH,
        "Checkpoints": CHECKPOINT_DIR,
    }
    
    all_ok = True
    for name, path in paths.items():
        exists = path.exists()
        status = "✓" if exists else "✗ (missing)"
        print(f"  {name}: {path}")
        print(f"    Status: {status}")
        if not exists:
            all_ok = False
    
    # Check MathWriting contents
    if MATHWRITING_PATH.exists():
        synthetic = MATHWRITING_PATH / "synthetic"
        bboxes = MATHWRITING_PATH / "synthetic-bboxes.jsonl"
        symbols = MATHWRITING_PATH / "symbols"
        
        print(f"\n  MathWriting Contents:")
        print(f"    synthetic/: {'✓' if synthetic.exists() else '✗'}")
        print(f"    synthetic-bboxes.jsonl: {'✓' if bboxes.exists() else '✗'}")
        print(f"    symbols/: {'✓' if symbols.exists() else '✗'}")
        
        if synthetic.exists():
            inkml_count = len(list(synthetic.glob("**/*.inkml")))
            print(f"    InkML files: {inkml_count:,}")
    
    print("=" * 50)
    return all_ok


if __name__ == "__main__":
    check_data_paths()

