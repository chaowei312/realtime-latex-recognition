# Stroke Capture Tool - Annotator Guide

## Overview

This is a web-based tool for collecting handwritten mathematical symbol stroke data. Your task is to draw multiple handwritten variants for each mathematical symbol, which will be used to train an AI model to recognize handwritten LaTeX formulas.

## Getting Started

1. Double-click to open `capture_tool.html` (opens in browser)
2. Select the symbol to draw
3. Draw the symbol on the canvas
4. Press **Enter** to save
5. Repeat to draw more variants
6. When finished, click **Download** to download the JSON file

---

## Detailed Instructions

### Step 1: Select a Symbol

1. Use the dropdown menu on the right to select a symbol category:
   - **Uppercase Letters (A-Z)**
   - **Lowercase Letters (a-z)**
   - **Digits (0-9)**
   - **Greek Lowercase** (α, β, π...)
   - **Greek Uppercase** (Π, Σ, Ω...)
   - **Arithmetic** (+, -, ×, ÷...)
   - **Relations** (=, ≠, ≤, ⊂...)
   - **Calculus** (∫, ∂, ∑...)
   - **Other categories...**

2. Click the symbol button you want to draw
   - 🟢 Green = has existing data
   - 🔵 Blue border = currently selected

### Step 2: Draw the Symbol

1. Draw the symbol on the white canvas using your mouse
2. **Each stroke** (mouse down to mouse up) is automatically numbered as stroke0, stroke1, stroke2...
3. Different strokes are displayed in different colors

**Important Notes:**
- ⚠️ **Keep stroke order consistent each time you draw the same symbol!**
- Example: Letter "A" should always be drawn with left diagonal first, then right diagonal, then the horizontal bar

### Step 3: Save

- Press the **Enter** key to save
- Or click the **Save (Enter)** button

### Step 4: Draw More Variants

- Draw **3-5 variants** for each symbol
- Variants can have stylistic differences (e.g., some rounder, some more angular)
- But stroke order must remain consistent

---

## Keyboard Shortcuts

| Key | Function |
|-----|----------|
| **Enter** | Save current drawing |
| **Z** | Undo last stroke |
| **C** | Clear canvas |

---

## Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Left: Canvas Area                Right: Control Panel      │
├─────────────────────────────────────────────────────────────┤
│                                   1. Symbol Selection       │
│   ┌─────────────────┐               [Dropdown] [Grid]       │
│   │                 │                                       │
│   │   Draw Here     │            2. Saved Data              │
│   │                 │               Shows variation count   │
│   │                 │               for each stroke         │
│   └─────────────────┘                                       │
│                                   3. JSON Output            │
│   [Save] [Undo] [Clear] [Download]   Live data preview      │
│                                                             │
│   Stats: Symbols | Variations | Combinations                │
└─────────────────────────────────────────────────────────────┘
```

---

## Annotation Priority

Please annotate in the following priority order:

### High Priority (Must Complete)
1. **A-Z** Uppercase letters (26)
2. **a-z** Lowercase letters (26)
3. **0-9** Digits (10)
4. **Common Greek lowercase**: α, β, γ, δ, θ, λ, μ, π, σ, φ, ω
5. **Common Greek uppercase**: Γ, Δ, Σ, Π, Ω

### Medium Priority
6. **Basic operators**: +, -, ×, ÷, =, ≠
7. **Calculus symbols**: ∫, ∂, ∑, ∏, √, ∞
8. **Relation symbols**: <, >, ≤, ≥, ⊂, ∈

### Low Priority
9. Other symbols

---

## Guidelines

### ✅ Best Practices

- Keep stroke order consistent
- Draw 3-5 variants per symbol
- Write naturally, don't be overly deliberate
- Personal style variations are welcome

### ❌ Common Mistakes

- Inconsistent stroke order
- Drawing only 1 variant
- Drawing too small or too large (aim for 50%-80% of canvas)
- Saving without undoing mistakes

---

## Special Symbol Notes

### Circle Modifier ○

This is a special symbol used for:
- Standalone: represents a circle
- Combined with integral: ∫ + ○ = ∮ (contour integral)

**Use case**: When a user draws a circle on an integral symbol, the system recognizes it as a contour integral.

### Composite Symbols

The following symbols don't need separate annotation - they are automatically composed by AI:
- **∮** = ∫ + ○ (contour integral)
- **∛** = √ + 3 (cube root = square root + 3)
- **″** = ′ + ′ (double prime = two single primes)

---

## Data Submission

1. After completing annotation, click **Download** to download `stroke_data.json`
2. Send the file to the project lead
3. Data is automatically saved in browser local storage, progress is restored on next visit

---

## FAQ

**Q: What if I make a mistake?**
A: Press Z to undo the last stroke, or press C to clear and redraw

**Q: Will I lose data if I close the browser?**
A: No, data is automatically saved in browser local storage

**Q: How do I delete all data for a symbol?**
A: Select the symbol, then click the 🗑 button next to the title

**Q: How many variants do I need per symbol?**
A: Recommended 3-5, more is better (more variants = more combinations = better AI training)

**Q: Can I use a drawing tablet?**
A: Yes! The tool supports touchscreens and drawing tablets

---

## Progress Statistics

The interface footer displays:
- **Symbols**: Number of annotated symbols
- **Variations**: Total number of variants
- **Combinations**: Possible combinations (variants multiplied)

Goals:
- High priority symbols (~80) × 3 variants = 240 handwritten samples
- Ideal: 5 variants per symbol = 400 handwritten samples

---

## Contact

For questions, please contact the project lead.

Thank you for your contribution!
