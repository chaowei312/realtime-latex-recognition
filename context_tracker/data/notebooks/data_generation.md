# Data Generation Plan: Handwritten Math Symbols & Commutative Diagrams

## Overview

This document outlines the plan for generating synthetic handwritten mathematical expressions and commutative diagrams for training the stroke-level LaTeX editing model.

---

## 1. Generation Approaches Overview

| Approach | Realism | Speed | Model Size | Edge Deploy |
|----------|---------|-------|------------|-------------|
| **Template + Noise** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 0 MB | ✅ |
| **RNN (Graves)** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~5 MB | ✅ |
| **Diffusion (One-DM)** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ~500 MB | ⚠️ |
| **Flow Matching** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~200 MB | ⚠️ |

---

## 2. Neural Network Models for Realistic Handwriting

### 2.1 Diffusion-Based Models (Best Quality)

#### One-DM (One-Shot Diffusion Mimicker) ⭐⭐⭐⭐⭐
- **Paper**: "One-DM: One-Shot Diffusion Mimicker for Handwritten Text Generation"
- **GitHub**: https://github.com/dailenson/One-DM
- **Features**: 
  - Single reference sample → generate any text in that style
  - High-frequency style extraction
  - Multi-language support
- **Size**: ~500MB
- **Use case**: Generate realistic math symbols from single example

```python
# Example usage (One-DM)
from one_dm import OneDM

model = OneDM.from_pretrained("dailenson/one-dm")
style_image = load_image("reference_handwriting.png")
output = model.generate(text="∫", style_ref=style_image)
```

#### DiffusionPen (Few-Shot) ⭐⭐⭐⭐
- **Paper**: "DiffusionPen: Towards Controlling the Style of Handwritten Text Generation"
- **GitHub**: https://github.com/koninik/DiffusionPen
- **Features**:
  - 5 reference samples → learn writer style
  - Metric learning + classification
  - ECCV 2024
- **Size**: ~400MB

#### Diffusion-Handwriting-Generation ⭐⭐⭐⭐
- **GitHub**: https://github.com/tcl9876/Diffusion-Handwriting-Generation
- **Features**:
  - Online (stroke) and offline (image) generation
  - Trainable on custom data
  - Good for math symbols

### 2.2 Lightweight Models (For Edge/Mobile)

#### Graves Handwriting RNN ⭐⭐⭐
- **Paper**: "Generating Sequences With Recurrent Neural Networks" (2013)
- **GitHub**: https://github.com/sjvasquez/handwriting-synthesis
- **Features**:
  - Very lightweight (~5MB)
  - Generates stroke sequences directly
  - Fast inference
- **Limitation**: Text only, not math symbols out-of-box

```python
# Example usage (Graves RNN)
from handwriting_synthesis import Hand

hand = Hand()
strokes = hand.write("Hello", style=7)  # Returns stroke coordinates
```

#### BK-SDM (Lightweight Stable Diffusion) ⭐⭐⭐
- **Paper**: "BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion"
- **Features**:
  - 30-50% size reduction
  - Faster inference
  - Can use ControlNet for sketch guidance
- **GitHub**: Based on CompVis/stable-diffusion

### 2.3 Flow Matching Models (Emerging)

#### Rectified Flow ⭐⭐⭐⭐
- **Features**:
  - Faster than diffusion (fewer steps)
  - Straight trajectories → easier training
  - Good for stroke generation
- **Status**: Research stage for handwriting

### 2.4 ControlNet for Sketch-to-Handwriting

Use pretrained ControlNet to convert LaTeX→rendered→handwriting:

```
LaTeX "x^2" → Render to image → ControlNet Scribble → Handwritten style
```

- **Model**: lllyasviel/sd-controlnet-scribble
- **HuggingFace**: https://huggingface.co/lllyasviel/sd-controlnet-scribble

---

## 3. Recommended Pipeline for Our Project

### Option A: Quick Start (TikZ-CD + Template-based)
```
1. Use generate_tikzcd.py for diagrams & expressions
2. Add noise/perturbation for variation
3. Generate 10K+ samples quickly
```
**Pros**: Fast, no GPU needed, reproducible, high-quality math rendering
**Cons**: Less realistic handwriting style

See: `tikzcd_editing_demo.ipynb` for examples

### Option B: Hybrid (Template + Neural Enhancement)
```
1. Generate base strokes with templates
2. Use Graves RNN to add style variation
3. Apply neural texture/distortion
```
**Pros**: Balance of speed and realism
**Cons**: Requires model training

### Option C: Full Neural (Diffusion)
```
1. Fine-tune One-DM on math symbols
2. Collect few real handwritten samples per symbol
3. Generate unlimited variations
```
**Pros**: Most realistic
**Cons**: GPU required, slower generation

---

## 4. Basic Python Libraries (Non-Neural)

### 4.1 Python Libraries

| Library | Purpose | Installation | Notes |
|---------|---------|--------------|-------|
| **svgwrite** | Create SVG paths | `pip install svgwrite` | Good for vector strokes |
| **bezier** | Bezier curve manipulation | `pip install bezier` | Smooth stroke interpolation |
| **noise** | Perlin noise for perturbation | `pip install noise` | Natural-looking distortion |
| **scipy.ndimage** | Elastic deformation | `pip install scipy` | Image-level augmentation |
| **PIL/Pillow** | Image rendering | `pip install pillow` | Rasterization |
| **matplotlib** | Visualization & rendering | `pip install matplotlib` | Quick prototyping |
| **cairosvg** | SVG to image | `pip install cairosvg` | High-quality rendering |

### 4.2 Template-Based Generation Approaches

1. **Bezier Curve + Noise**: Define symbol as control points, add Perlin noise
2. **Stroke Perturbation**: Take template strokes, apply random transformations
3. **Font-based + Distortion**: Render with handwriting font, apply elastic deform
4. **RNN/VAE Synthesis**: Use trained models (heavier, but more realistic)

---

## 2. Symbol Generation Strategy

### 2.1 Basic Symbols (Template-based)

For simple symbols, define template strokes and add variations:

```python
# Template for "x" - two crossing strokes
X_TEMPLATE = {
    "strokes": [
        [(0, 0), (1, 1)],      # diagonal /
        [(0, 1), (1, 0)],      # diagonal \
    ],
    "variations": {
        "rotation": (-15, 15),      # degrees
        "scale": (0.8, 1.2),
        "stroke_noise": 0.05,       # relative to size
        "point_jitter": 0.02,
    }
}
```

### 2.2 Complex Symbols (Bezier-based)

For curved symbols (∫, ∑, α, etc.), use Bezier curves:

```python
# Template for integral sign "∫"
INTEGRAL_TEMPLATE = {
    "bezier_curves": [
        # Control points for the S-curve
        [(0.3, 0), (0.1, 0.3), (0.5, 0.5), (0.2, 0.7), (0.4, 1.0)]
    ],
    "variations": {...}
}
```

### 2.3 Greek Letters

| Letter | Template Type | Complexity |
|--------|---------------|------------|
| α (alpha) | Bezier (loop + tail) | Medium |
| β (beta) | Bezier (two loops) | Medium |
| γ (gamma) | Bezier (curved) | Medium |
| θ (theta) | Ellipse + line | Easy |
| π (pi) | Two strokes | Easy |
| σ (sigma) | Bezier (loop) | Medium |
| Σ (Sigma) | Three lines | Easy |

---

## 3. Arrow Generation (for Commutative Diagrams)

### 3.1 Arrow Types

| Arrow | LaTeX | Strokes | Template |
|-------|-------|---------|----------|
| → | `\to` | Line + arrowhead | 1 line + 2 short lines |
| ⇒ | `\Rightarrow` | Double line + head | 2 parallel + head |
| ↦ | `\mapsto` | Line + head + tail bar | 3 strokes |
| ↪ | `\hookrightarrow` | Hook + line + head | Curve + line + head |
| ↠ | `\twoheadrightarrow` | Line + double head | Line + 4 short lines |

### 3.2 Arrow Generation Code Structure

```python
def generate_arrow(start, end, arrow_type="simple", noise_level=0.05):
    """Generate handwritten arrow from start to end point."""
    # 1. Calculate direction and length
    # 2. Generate main shaft with noise
    # 3. Add arrowhead strokes
    # 4. Apply global transformations
    return strokes
```

---

## 4. Commutative Diagram Generation (TikZ-CD)

**Use TikZ-CD** - the standard LaTeX package for commutative diagrams.

### 4.1 TikZ-CD Templates (see generate_tikzcd.py)

```python
from generate_tikzcd import TikzCDGenerator

# Simple arrow: A --f--> B
TikzCDGenerator.simple_arrow()
# Output: \begin{tikzcd} A \arrow[r, "f"] & B \end{tikzcd}

# Commutative triangle
TikzCDGenerator.triangle()
# Output: \begin{tikzcd}
#         A \arrow[r, "f"] \arrow[dr, "h"'] & B \arrow[d, "g"] \\
#         & C
#         \end{tikzcd}

# Commutative square
TikzCDGenerator.square()
# Output: \begin{tikzcd}
#         A \arrow[r, "f"] \arrow[d, "g"'] & B \arrow[d, "h"] \\
#         C \arrow[r, "k"'] & D
#         \end{tikzcd}

# Pullback (with corner symbol)
TikzCDGenerator.pullback()

# Pushout
TikzCDGenerator.pushout()

# Exact sequence
TikzCDGenerator.exact_sequence()

# Double arrow (parallel morphisms)
TikzCDGenerator.double_arrow()

# Natural transformation (with 2-cell)
TikzCDGenerator.natural_transformation()
```

### 4.2 Available Diagram Types

| Diagram | TikZ-CD | Description |
|---------|---------|-------------|
| Simple arrow | `A \arrow[r] & B` | Basic morphism |
| Triangle | 3 nodes, 3 arrows | g ∘ f = h |
| Square | 4 nodes, 4 arrows | h ∘ f = k ∘ g |
| Pullback | Square + ⌟ marker | Universal property |
| Pushout | Square + ⌜ marker | Universal property |
| Exact sequence | Linear chain | 0 → A → B → C → 0 |
| Double arrow | Parallel arrows | Equalizer/coequalizer |
| Adjunction | L ⊣ R | Adjoint functors |
| Natural transformation | 2-cell arrow | η: F ⇒ G |

### 4.3 Rendering Pipeline

```
TikZ-CD Code → pdflatex → PDF → pdf2image → PNG
                                    ↓
                        (Optional) Neural style transfer
                                    ↓
                            Handwritten style PNG
```

### 4.4 Requirements

```bash
# LaTeX (MiKTeX or TexLive)
# Windows: https://miktex.org/download
# Linux: sudo apt install texlive-full

# Python dependencies
pip install pdf2image

# Poppler (for pdf2image)
# Windows: conda install -c conda-forge poppler
# Linux: sudo apt install poppler-utils
```

### 4.5 Usage

```python
from generate_tikzcd import TikzCDGenerator, render_tikzcd_to_image

# Generate tikz code
code = TikzCDGenerator.square(A="X", B="Y", C="Z", D="W")

# Render to image
render_tikzcd_to_image(code, "square.png", dpi=200)
```

---

## 5. Data Format (InkML-compatible)

### 5.1 Output Structure

```
data/
├── generated/
│   ├── symbols/
│   │   ├── x_001.inkml
│   │   ├── alpha_001.inkml
│   │   └── ...
│   ├── arrows/
│   │   ├── arrow_simple_001.inkml
│   │   └── ...
│   └── diagrams/
│       ├── triangle_001.inkml
│       ├── square_001.inkml
│       └── ...
├── metadata.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 5.2 InkML Template

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ink xmlns="http://www.w3.org/2003/InkML">
  <trace id="t1">x1 y1 t1, x2 y2 t2, ...</trace>
  <trace id="t2">...</trace>
  
  <traceGroup>
    <traceGroup xml:id="symbol_1">
      <annotation type="truth">x</annotation>
      <traceView traceDataRef="t1"/>
      <traceView traceDataRef="t2"/>
    </traceGroup>
  </traceGroup>
  
  <annotation type="truth">\to</annotation>
</ink>
```

---

## 6. Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. TEMPLATE DEFINITION                                              │
│     └── symbols.py: Define base templates (control points)           │
│                                                                      │
│  2. VARIATION GENERATION                                             │
│     └── augment.py: Add noise, rotation, scale, stroke width         │
│                                                                      │
│  3. STROKE RENDERING                                                 │
│     └── render.py: Convert to coordinates, interpolate               │
│                                                                      │
│  4. DIAGRAM COMPOSITION                                              │
│     └── diagrams.py: Combine symbols + arrows in 2D layout           │
│                                                                      │
│  5. OUTPUT GENERATION                                                │
│     ├── to_inkml(): Save as InkML format                             │
│     ├── to_image(): Render to PNG for visualization                  │
│     └── to_json(): Save metadata                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Target Dataset Size

| Category | Count | Variations | Total |
|----------|-------|------------|-------|
| Basic symbols (a-z, 0-9) | 36 | 100 each | 3,600 |
| Greek letters | 24 | 100 each | 2,400 |
| Operators (+, -, ×, etc.) | 20 | 100 each | 2,000 |
| Arrows (6 types) | 6 | 200 each | 1,200 |
| Simple diagrams | 3 | 500 each | 1,500 |
| Complex diagrams | 5 | 200 each | 1,000 |
| **Total** | | | **~12,000** |

---

## 8. Implementation Order

- [ ] Phase 1: Basic symbol generation (lines, curves)
- [ ] Phase 2: Greek letter templates
- [ ] Phase 3: Arrow generation
- [ ] Phase 4: Diagram composition
- [ ] Phase 5: InkML export
- [ ] Phase 6: Visualization & validation
- [ ] Phase 7: Large-scale generation

---

## 9. Quality Validation

1. **Visual inspection**: Random samples rendered and checked
2. **Stroke statistics**: Compare to real CROHME data (points/stroke, duration)
3. **Recognition test**: Run through existing HMER model
4. **Diversity check**: Ensure variations are sufficiently different

---

## 10. Model Size Comparison for Edge Deployment

| Model | Parameters | Size | GPU Required | Inference Time |
|-------|------------|------|--------------|----------------|
| **Template-based** | 0 | 0 MB | ❌ | <1ms |
| **Graves RNN** | ~3M | 5 MB | ❌ | ~10ms |
| **BK-SDM-Tiny** | ~100M | 400 MB | ⚠️ (CPU slow) | ~2s (GPU) |
| **One-DM** | ~200M | 500 MB | ✅ | ~1s (GPU) |
| **ControlNet + SD** | ~1.5B | 6 GB | ✅ | ~5s (GPU) |
| **FLUX.1-Distilled** | ~500M | 2 GB | ✅ | ~3s (GPU) |

### Recommendation for This Project

**For Data Generation (offline, quality matters)**:
- Use **One-DM** or **DiffusionPen** for highest quality
- GPU server for batch generation

**For Edge Deployment (real-time)**:
- Use **Template-based** with neural style transfer post-processing
- Or quantized **Graves RNN** for stroke generation

---

## 11. Quick Start: Using Existing Models

### Install One-DM
```bash
git clone https://github.com/dailenson/One-DM
cd One-DM
pip install -r requirements.txt

# Download pretrained weights
wget https://github.com/dailenson/One-DM/releases/download/v1.0/one_dm.pth
```

### Install Graves Handwriting
```bash
pip install handwriting-synthesis

# Usage
from handwriting_synthesis import Hand
hand = Hand()
lines = hand.write("x^2 + y = 5")  # Returns stroke coordinates
```

### Install DiffusionPen
```bash
git clone https://github.com/koninik/DiffusionPen
cd DiffusionPen
pip install -r requirements.txt
```

---

## 12. Future: Custom Math Symbol Model

To train a model specifically for math symbols:

1. **Collect Data**:
   - Use CROHME InkML (real handwritten math)
   - Generate template-based samples
   - Mix both for training

2. **Fine-tune One-DM**:
   ```python
   # Fine-tune on math symbols
   python train.py \
       --data_path ./math_symbols/ \
       --pretrained one_dm.pth \
       --output math_one_dm.pth
   ```

3. **Generate Dataset**:
   ```python
   # Generate 100 variations per symbol
   for symbol in ["x", "y", "alpha", "integral", ...]:
       for i in range(100):
           img = model.generate(symbol, style_ref=random_style())
           save_inkml(img, f"{symbol}_{i:03d}.inkml")
   ```

---

*Last updated: December 2024*
