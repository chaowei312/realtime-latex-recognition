# Data Augmentation: Stroke-Level Training Data Generation

## Overview

This document describes the data generation and augmentation pipeline for the stroke-level LaTeX editing model. The approach uses **per-stroke variations** with **combinatorial augmentation** to generate diverse training data from a small set of manually captured strokes.

---

## 1. Architecture: Stroke-Level Encoding

### 1.1 Design Decision: Parallel 2D Conv Along Stroke Trajectory

Instead of bounding box crops divided into patches, we extract **conv tokens along the stroke trajectory**:

```
User draws stroke: (x₁,y₁) → (x₂,y₂) → ... → (xₙ,yₙ)

Sample points along trajectory:

    Full canvas with stroke:
    ┌────────────────────────────┐
    │         ╱╲                 │
    │        ╱  ╲                │
    │       ╱────╲               │
    └────────────────────────────┘
    
    Extract local patches at sampled points:
    
    @(x₁,y₁)      @(x₂,y₂)      @(x₃,y₃)      @(x₄,y₄)
    ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
    │  ╱  │       │ ╱╲  │       │  ╲  │       │╱────│
    │ ╱   │       │╱  ╲ │       │   ╲ │       │     │
    └─────┘       └─────┘       └─────┘       └─────┘
       ↓             ↓             ↓             ↓
     Conv2D        Conv2D        Conv2D        Conv2D  (parallel batch)
       ↓             ↓             ↓             ↓
    [t₁,pos₁]    [t₂,pos₂]    [t₃,pos₃]    [t₄,pos₄]
```

### 1.2 Why This Approach?

| Aspect | Bounding Box Patches | Sliding Conv |
|--------|---------------------|--------------|
| **Background waste** | Many empty patches | Only stroke pixels processed |
| **Position preserved** | Requires extra embedding | Each token has (x,y) naturally |
| **Multi-stroke context** | Separate per stroke | Conv sees overlapping strokes |
| **Diagram support** | Hard to represent arrows | Arrows have shape + position |
| **Efficiency** | O(patches) | O(sampled points) |

### 1.3 Implementation

```python
class ParallelStrokeEncoder(nn.Module):
    """Parallel 2D conv along stroke trajectory."""
    
    def __init__(self, patch_size=16, embed_dim=256, stride=16):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride  # Sample every N pixels along stroke
        
        # Small 2D conv for local patch features
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 16x16 → 4x4
            nn.Flatten(),
            nn.Linear(64 * 16, embed_dim)
        )
    
    def forward(self, canvas, stroke_points):
        """
        Args:
            canvas: [H, W] full drawing canvas
            stroke_points: [N, 2] points along stroke trajectory
        Returns:
            tokens: [M, embed_dim] conv tokens
            positions: [M, 2] (x, y) positions for 2D RoPE
        """
        # 1. Sample positions (not every pixel)
        positions = stroke_points[::self.stride]  # [M, 2]
        
        # 2. Batch extract ALL patches at once (GPU parallel)
        patches = self.batch_extract(canvas, positions)  # [M, 1, P, P]
        
        # 3. Single forward pass through conv (GPU parallel)
        tokens = self.conv(patches)  # [M, embed_dim]
        
        return tokens, positions
    
    def batch_extract(self, canvas, positions):
        """Extract all patches in parallel using grid_sample."""
        M = len(positions)
        P = self.patch_size
        
        # Create sampling grids for all patches
        grids = self.create_patch_grids(positions, P, canvas.shape)
        
        # Batch sample - single GPU operation
        canvas_batched = canvas.unsqueeze(0).unsqueeze(0).expand(M, 1, -1, -1)
        patches = F.grid_sample(canvas_batched, grids, align_corners=True)
        
        return patches  # [M, 1, P, P]
```

### 1.4 Key Benefits

1. **Local Visual Context**: Each token sees surrounding strokes (helps with 't' crossbar, '+' recognition)
2. **Position Preserved**: Every token has (x, y) for 2D RoPE attention
3. **Scale Awareness**: Patch size relative to stroke reveals size information
4. **Diagram Support**: Arrows encoded with shape + endpoints preserved

---

## 2. Data Collection: Stroke Capture Tool

### 2.1 Tool Overview

Manual stroke capture via web-based tool (`stroke_corpus/capture_tool.html`):

```
┌──────────────────────────────────────────────────────────────────┐
│  STROKE CAPTURE TOOL                                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Category Dropdown] [Symbol Buttons: A B C α β γ ∫ ∑ ...]       │
│                                                                   │
│  ┌─────────────────────────────────────────┐                     │
│  │                                         │                     │
│  │           Drawing Canvas                │                     │
│  │       (500 × 500 pixels)               │                     │
│  │                                         │                     │
│  │         User draws strokes              │                     │
│  │         Pen press → Pen lift            │                     │
│  │                                         │                     │
│  └─────────────────────────────────────────┘                     │
│                                                                   │
│  Strokes: stroke0 ✓  stroke1 ✓  stroke2 (drawing...)            │
│                                                                   │
│  [Save Strokes] [Clear] [Delete Symbol]                          │
│                                                                   │
│  Saved Variations: [■] [■] [■]  (thumbnails)                     │
│                                                                   │
│  [Download JSON]                                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Workflow

1. **Select Symbol**: Choose category (e.g., "Greek-Lower") and symbol (e.g., "α")
2. **Draw Strokes**: Draw all strokes for the symbol naturally
3. **Save Variation**: Click "Save Strokes" to save as one variation
4. **Repeat**: Draw 3-5 variations of the same symbol
5. **Download**: Export all data as JSON

### 2.3 Supported Symbol Categories (238 total)

| Category | Count | Examples |
|----------|-------|----------|
| Digits | 10 | 0, 1, 2, ..., 9 |
| Latin-Lower | 26 | a, b, c, ..., z |
| Latin-Upper | 26 | A, B, C, ..., Z |
| Greek-Lower | 24 | α, β, γ, δ, ..., ω |
| Greek-Upper | 11 | Γ, Δ, Θ, Λ, ..., Ω |
| Operators | 20 | +, −, ×, ÷, ±, ... |
| Relations | 18 | =, ≠, <, >, ≤, ≥, ... |
| Arrows | 13 | →, ←, ↔, ⇒, ⇐, ... |
| Brackets | 12 | (, ), [, ], {, }, ... |
| Calculus | 15 | ∫, ∂, ∇, ∑, ∏, ... |
| Sets | 12 | ∈, ∉, ⊂, ⊃, ∪, ∩, ... |
| Logic | 8 | ∧, ∨, ¬, ⊢, ⊨, ... |
| Modifiers | 17 | ˙, ¨, ^, ˜, ′, ... |
| Category-Theory | 9 | ⟲, ⟳, ○, ◁, ▷, ... |

### 2.4 Compositional Symbols

Some symbols are compositions of primitives:

| Symbol | Composition | Note |
|--------|-------------|------|
| ∮ | ∫ + ○ | Circle modifier on integral |
| ∛ | √ + 3 | Radical + number above |
| ″ | ′ + ′ | Two prime marks |

The model learns spatial relationships implicitly from patch token distributions.

---

## 3. Data Format: Per-Stroke Variations

### 3.1 JSON Structure

```json
{
  "A": {
    "symbol": "A",
    "latex": "A",
    "strokes": {
      "stroke0": {
        "variations": [
          [[0.584, 0.0], [0.211, 0.476], [0.0, 1.0]],
          [[0.587, 0.0], [0.0, 1.0]],
          [[0.433, 0.022], [0.0, 0.787]]
        ]
      },
      "stroke1": {
        "variations": [
          [[0.616, 0.017], [0.724, 0.162], [0.827, 0.605], [1.0, 0.821]],
          [[0.587, 0.027], [0.923, 0.546], [1.0, 0.752]],
          [[0.423, 0.0], [1.0, 1.0]]
        ]
      },
      "stroke2": {
        "variations": [
          [[0.319, 0.443], [0.703, 0.443], [0.762, 0.466]],
          [[0.361, 0.573], [0.861, 0.443]],
          [[0.201, 0.596], [0.758, 0.55]]
        ]
      }
    }
  }
}
```

### 3.2 Key Design Choices

| Choice | Rationale |
|--------|-----------|
| **Per-stroke variations** | Combinatorial: 3×3×3 = 27 ways to write 'A' |
| **Global normalization** | All strokes normalized to symbol's bounding box |
| **Normalized coordinates** | [0,1] range for resolution independence |
| **Point sequences** | Efficient for GPU rendering |

### 3.3 Global Normalization

All strokes for a symbol are normalized relative to the **entire symbol's bounding box**:

```
Drawing: User draws full symbol naturally
         ┌────────────────┐
         │    ╱╲          │
         │   ╱  ╲         │  Global bbox covers
         │  ╱────╲        │  ALL strokes
         │ ╱      ╲       │
         └────────────────┘

Normalization: Each stroke's points normalized to [0,1] within global bbox
- stroke0 (left leg):  preserved relative position
- stroke1 (right leg): preserved relative position  
- stroke2 (crossbar):  preserved relative position

Composition: When rendering, all strokes assemble correctly
             because they share the same coordinate space
```

---

## 4. Combinatorial Augmentation

### 4.1 Per-Stroke Variation Mixing

With 3 variations per stroke and 3 strokes:

```
stroke0:  var0 ──┐
          var1 ──┼──┐
          var2 ──┘  │
                    ├─► 3 × 3 × 3 = 27 unique combinations
stroke1:  var0 ──┐  │
          var1 ──┼──┤
          var2 ──┘  │
                    │
stroke2:  var0 ──┐  │
          var1 ──┼──┘
          var2 ──┘
```

### 4.2 Rendering Pipeline

```python
def render_symbol_variation(symbol_data, variation_indices):
    """
    Render one combination of stroke variations.
    
    Args:
        symbol_data: JSON data for symbol
        variation_indices: [i, j, k] - which variation for each stroke
    Returns:
        canvas: Rendered image
        all_points: All stroke points for conv encoding
    """
    canvas = np.zeros((256, 256))
    all_points = []
    
    for stroke_idx, (stroke_name, stroke_data) in enumerate(symbol_data['strokes'].items()):
        var_idx = variation_indices[stroke_idx]
        points = stroke_data['variations'][var_idx]
        
        # Scale from [0,1] to canvas size
        scaled_points = [(p[0] * 255, p[1] * 255) for p in points]
        
        # Draw stroke
        draw_stroke(canvas, scaled_points, line_width=3)
        
        all_points.extend(scaled_points)
    
    return canvas, all_points
```

### 4.3 Additional Augmentations

| Augmentation | Range | Purpose |
|--------------|-------|---------|
| **Jitter** | ±2-5 pixels | Natural hand tremor |
| **Rotation** | ±5-10° | Writing angle variation |
| **Scale** | 0.9-1.1× | Size variation |
| **Thickness** | 1-4 pixels | Pen pressure variation |
| **Speed noise** | Bezier smoothing | Fast vs slow writing |

### 4.4 GPU-Accelerated Rendering

```python
class GPUStrokeRenderer:
    """Batched GPU rendering with PyTorch + kornia."""
    
    def __init__(self, canvas_size=256, device='cuda'):
        self.size = canvas_size
        self.device = device
    
    def render_batch(self, stroke_batch, thickness=3.0):
        """
        Render multiple symbols in parallel.
        
        Args:
            stroke_batch: List of stroke point lists
            thickness: Line thickness
        Returns:
            canvases: [B, 1, H, W] tensor
        """
        B = len(stroke_batch)
        canvases = torch.zeros(B, 1, self.size, self.size, device=self.device)
        
        for b, strokes in enumerate(stroke_batch):
            for stroke_points in strokes:
                # Convert to tensor
                points = torch.tensor(stroke_points, device=self.device)
                
                # Render using distance field (anti-aliased)
                canvas = self.render_stroke_distance_field(points, thickness)
                canvases[b, 0] += canvas
        
        return torch.clamp(canvases, 0, 1)
    
    def render_stroke_distance_field(self, points, thickness):
        """Render stroke using signed distance field for smooth anti-aliasing."""
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(self.size, device=self.device),
            torch.arange(self.size, device=self.device),
            indexing='ij'
        )
        
        # Compute distance to each line segment
        min_dist = torch.full((self.size, self.size), float('inf'), device=self.device)
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dist = self.point_to_segment_distance(x, y, p1, p2)
            min_dist = torch.minimum(min_dist, dist)
        
        # Convert distance to intensity (smooth falloff)
        intensity = torch.clamp(1.0 - min_dist / thickness, 0, 1)
        
        return intensity
```

---

## 5. Training Data Generation

### 5.1 Example Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING EXAMPLE GENERATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SELECT SYMBOL                                                │
│     └── Pick from 238 available symbols                         │
│                                                                  │
│  2. SELECT VARIATION COMBINATION                                 │
│     └── Random [i, j, k, ...] indices for each stroke           │
│                                                                  │
│  3. RENDER TO CANVAS                                             │
│     └── GPU batch render with augmentations                     │
│                                                                  │
│  4. EXTRACT CONV TOKENS                                          │
│     └── Sample along stroke trajectory                          │
│     └── Apply 2D conv to each patch                             │
│     └── Keep (x, y) positions                                   │
│                                                                  │
│  5. GENERATE TRAINING PAIR                                       │
│     ├── Input: conv tokens + positions + context                │
│     └── Output: LaTeX edit operation                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Stroke-Level Training Examples

```python
@dataclass
class StrokeLevelExample:
    """Single stroke-level training example."""
    
    # Input
    canvas: np.ndarray              # Full canvas with all strokes
    new_stroke_points: List[Tuple]  # Points of new stroke
    context_latex: str              # Previous LaTeX context
    
    # Extracted features
    conv_tokens: Tensor             # [M, embed_dim] from stroke conv
    positions: Tensor               # [M, 2] for 2D RoPE
    
    # Output
    operation: str                  # ADD, REPLACE, INSERT, WAIT
    target_latex: str               # Updated LaTeX
    
    # Metadata
    symbol: str                     # Symbol being drawn
    stroke_idx: int                 # Which stroke (for multi-stroke symbols)
    is_complete: bool               # True if symbol is complete
```

### 5.3 WAIT vs Symbol Output

For multi-stroke symbols, the model learns when to wait:

```
Drawing letter 'A' (3 strokes):

Stroke 1 (left leg):   → Output: <WAIT>  (incomplete)
Stroke 2 (right leg):  → Output: <WAIT>  (incomplete)  
Stroke 3 (crossbar):   → Output: "A"     (complete!)

The model learns:
- Recognize partial symbols
- Wait for completion
- Output symbol only when confident
```

---

## 6. Noise Augmentation

### 6.1 Stroke-Level Noise Types

| Type | Description | Purpose |
|------|-------------|---------|
| `stray_dot` | Random tap marks | Pen contact while thinking |
| `partial_neighbor` | Adjacent symbol stroke | Neighbor crossing into view |
| `edge_intrusion` | Stroke from edge | Diagonal slash entering |

### 6.2 Why Stroke-Level Noise?

```
Image-level noise:              Stroke-level noise:
┌──────────────┐                ┌──────────────┐
│  ░░░░░░░░░░░ │                │        •     │
│  ░░[strokes]░│                │    [strokes] │
│  ░░░░░░░░░░░ │                │    •         │
└──────────────┘                └──────────────┘
 Noise is pixels                Noise is strokes
 Model can't learn              Model learns to
 to ignore it                   give low attention
```

### 6.3 Integration with Conv Encoding

```python
def generate_noisy_example(symbol_data, noise_prob=0.5):
    """Generate example with stroke-level noise."""
    
    # 1. Render symbol strokes
    canvas, symbol_points = render_symbol(symbol_data)
    
    # 2. Maybe add noise strokes
    noise_points = []
    if random.random() < noise_prob:
        noise_type = random.choice(['stray_dot', 'partial_neighbor', 'edge_intrusion'])
        noise_strokes = generate_noise_strokes(noise_type, canvas.shape)
        
        for stroke in noise_strokes:
            draw_stroke(canvas, stroke)
            noise_points.extend(stroke)
    
    # 3. Extract conv tokens from ALL points (symbol + noise)
    all_points = symbol_points + noise_points
    conv_tokens, positions = extract_conv_tokens(canvas, all_points)
    
    # 4. Mark which tokens are noise (for attention learning)
    noise_mask = create_noise_mask(len(symbol_points), len(noise_points))
    
    return {
        'canvas': canvas,
        'conv_tokens': conv_tokens,
        'positions': positions,
        'noise_mask': noise_mask,  # Model learns low attention here
    }
```

---

## 7. Dataset Statistics

### 7.1 From Manual Capture

| Metric | Target |
|--------|--------|
| Symbols | 238 unique |
| Variations per stroke | 3-5 |
| Strokes per symbol | 1-5 (average ~2.5) |
| Combinations per symbol | 27-125 |
| Total unique renderings | ~50,000+ |

### 7.2 With Augmentation

| Augmentation | Multiplier |
|--------------|------------|
| Jitter variations | ×10 |
| Rotation | ×5 |
| Scale | ×3 |
| Thickness | ×3 |
| **Total multiplier** | ×450 |

**Final dataset size**: ~50,000 × 450 = **~22.5M training examples**

### 7.3 Operation Distribution

| Operation | Percentage | Source |
|-----------|------------|--------|
| ADD | 55% | Sequential building |
| REPLACE | 25% | Confusion pair substitution |
| INSERT | 10% | Sub-expression insertion |
| WAIT | 10% | Incomplete multi-stroke symbols |

---

## 8. File Structure

```
data/
├── stroke_corpus/
│   ├── capture_tool.html          # Web-based capture tool
│   ├── annotation_guidelines.md   # Labeler instructions
│   ├── merge_annotations.py       # Merge annotator data
│   └── annotations/               # Per-symbol stroke data
│       ├── stroke_data_merged.json
│       └── ...
├── stroke_primitives.py           # Training data structures
├── demos/
│   └── stroke_demo.py             # Demo and visualization
├── data_augmentation.md           # This document
└── compositional_augmentation.py  # Compositional augmentation
```

---

## 9. Usage

### 9.1 Capture New Symbols

1. Open `stroke_corpus/capture_tool.html` in browser
2. Select symbol category and symbol
3. Draw strokes naturally
4. Save 3-5 variations
5. Download JSON

### 9.2 Generate Training Data

```python
from stroke_primitives import StrokeLevelGenerator

generator = StrokeLevelGenerator(
    symbols_dir='data/stroke_corpus/annotations/',
    canvas_size=256,
    sample_stride=16,
)

# Generate batch of examples
examples = generator.generate_batch(
    batch_size=64,
    noise_probability=0.5,
    augment=True,
)

# Each example contains:
# - conv_tokens: [M, embed_dim]
# - positions: [M, 2] 
# - context_latex: str
# - target_output: str (symbol or <WAIT>)
```

### 9.3 Render Visualizations

```python
from stroke_primitives import render_all_variations

# Render all combinations of 'A'
render_all_variations(
    symbol_json='data/stroke_corpus/annotations/A.json',
    output_path='A_variations.png',
    grid_size=(3, 3),
)
```

---

## 10. Summary

| Component | Approach |
|-----------|----------|
| **Stroke Encoding** | Parallel 2D conv along trajectory |
| **Data Collection** | Web-based manual capture tool |
| **Data Format** | Per-stroke variations, globally normalized |
| **Augmentation** | Combinatorial mixing + jitter/rotation/scale |
| **Noise** | Stroke-level (not image-level) |
| **Scale** | 238 symbols × 27+ variations × 450 augmentation = ~22M examples |

This approach enables:
- ✅ Efficient training data generation from small manual effort
- ✅ Natural variation through combinatorial mixing
- ✅ Position-aware encoding for 2D RoPE attention
- ✅ Robust noise handling via attention learning
- ✅ Diagram support through preserved spatial relationships

---

*Last updated: December 2024*
