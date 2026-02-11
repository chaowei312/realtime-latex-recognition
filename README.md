# Real-time LaTeX Recognition

Real-time stroke-level LaTeX recognition with incremental editing and reasoning trajectory capture.

## Overview

Traditional approaches to mathematical expression recognition process full images with O(N^2 x P^2) attention, producing a single output with no incremental updates. This project takes a fundamentally different approach:

- **Stroke-level tracking**: Process handwriting strokes as they happen, not frame-by-frame
- **Incremental LaTeX generation**: Build LaTeX output progressively, editing in-place rather than re-recognizing
- **Reasoning trajectory capture**: Record the model's decision process for interpretability

## Architecture

The system consists of three main components:

### 1. Context Tracker
The core module that maintains a running representation of the mathematical expression being written. Instead of full self-attention over all frames, it uses an incremental approach that only processes new stroke information against existing context.

### 2. Image-Text Demo
Visual encoder + decoder pipeline for converting handwritten strokes to LaTeX tokens, with autoregressive trajectory generation.

### 3. Training Dashboard
Web-based monitoring interface for tracking training progress in real-time.

## Key Differentiators vs Traditional OCR

| Aspect | Traditional VLM/OCR | This Project |
|--------|---------------------|--------------|
| Input | Static image | Streaming strokes |
| Processing | Full re-recognition each frame | Incremental updates |
| Complexity | O(N^2 x P^2) per inference | Grows only with new strokes |
| Output | Single LaTeX string | Edit operations (insert, delete, modify) |
| Interpretability | Black box | Reasoning trajectory captured |
| Latency | Grows with history | Constant for new strokes |

## Project Structure



## Project Status

Work in progress -- architecture designed, core modules under development.

## Tech Stack

- PyTorch
- Vision Transformers
- Autoregressive decoding
- Flask (training dashboard)

## License

MIT
