# Context Tracker: Minimal Prototype Plan

> **Goal**: Prove trajectory-based reasoning capture works in 2-3 weeks  
> **NOT**: Production-ready SOTA system

---

## 0. Key Innovation Demonstration

### Innovation vs Existing Methods

| Aspect | Existing Methods | **Ours** | Gain |
|--------|------------------|----------|------|
| **What's captured** | Final expression only | **Reasoning trajectory** (edit sequence) | New capability |
| **Visual processing** | All patches every step | **Current chunk only** → commit → discard | O(n³) → O(n²) |
| **Memory scaling** | O(n × p²) visual history | **O(n) text tokens** | ~100× reduction |
| **Edit operation** | Full re-inference | **O(1) local chunk re-embed** | ~10× faster |
| **Noise handling** | Clean input assumed | **Stroke artifacts** (model filters noise) | Robustness |

### Complexity: Why O(n³) → O(n²)

```
EXISTING: Keep all patches → Σ(ip)² = O(n³)
OURS:     Commit & discard → Σ(p² + i) = O(n²)

n=20, p=16:  Existing ~683K ops  vs  Ours ~5K ops  →  ~130× speedup
```

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  DUAL-HEAD MODEL: [VC] = Visual Class, [INT] = Integration                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  [LaTeX] ──causal──► [Patches] ──causal──► [VC] ─────────► [INT]               │
│                          │                   │                │                 │
│                          │                   └────────┬───────┘                 │
│                          │                      concat([INT],[VC])              │
│                          │                            │                         │
│                          │                     ┌──────┴──────┐                  │
│                          │                     │ QUERY PROJ  │                  │
│                          │                     │ W_q_h → q_h │                  │
│                          │                     └──────┬──────┘                  │
│                          │                            │                         │
│                   ┌──────┴──────┐                     │                         │
│                   │  KEY PROJ   │                     │                         │
│                   │ W_k_h → k_ij│                     │                         │
│                   └──────┬──────┘                     │                         │
│                          │                            │                         │
│                          └────────────┬───────────────┘                         │
│                                       ▼                                         │
│                          ┌─────────────────────────────┐                        │
│                          │ SCORE FIRST, THEN POOL      │                        │
│                          │                             │                        │
│                          │ Per-patch:  score_ij = q·k_ij                       │
│                          │ Per-stroke: score_i = mean(score_ij)                │
│                          │ Per-head:   repeat for h=1..4                       │
│                          │ Final:      mean across heads                       │
│                          │ mask = σ(final_score)                               │
│                          │                             │                        │
│                          │ HEAD B: Stroke Selection    │                        │
│                          └──────────────┬──────────────┘                        │
│                                         │                                       │
│  ┌────────────────────────┐             │                                       │
│  │ HEAD A: Recognition    │◄────────────┘                                       │
│  │ h_INT → FFN → P(sym)   │                                                     │
│  └────────────────────────┘                                                     │
│                                                                                  │
│  Attention: [VC] sees patches only, [INT] sees LaTeX + [VC]                    │
│  Head A: h_INT → FFN → symbol (context-aware recognition)                      │
│  Head B: Stroke Modification Module (SMM) - multi-head selection gate          │
│  Head C: Position Head - per-token edit position during AR                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Head C: Position Prediction (During AR Decoding)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  POSITION HEAD: Per-Token Edit Position (Each AR Step)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  At each AR step t, model outputs BOTH:                                         │
│    • Symbol: what token to insert (via Head A)                                 │
│    • Action: where to insert it (via Head C)                                   │
│                                                                                  │
│  Example: User writes "t" small, below "a" (subscript)                         │
│                                                                                  │
│    Step t:  h_t → (token="t", action=(a, SUB))                                 │
│             → Result: a_t  (t becomes subscript of a)                          │
│                                                                                  │
│  Key: User does NOT write "_" - it's LaTeX syntax!                             │
│       Model infers SUB relation from position of handwritten "t"               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Relationship Tensor (Parsing LaTeX to Structure)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LaTeX SYNTAX → PARSED INTO RELATIONS (not stored as tokens)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LaTeX String          Tokens (content only)      Relationship Tensor          │
│  ════════════════════  ════════════════════════   ══════════════════════════   │
│                                                                                  │
│  "a_t"                 ["a", "t"]                 (t -> a, SUB)                 │
│  "a^x"                 ["a", "x"]                 (x -> a, SUP)                 │
│  "\frac{a_t+b}{c}"     ["frac","a","t","+","b","c"]                             │
│                                                    (a -> frac, ABOVE)            │
│                                                    (c -> frac, BELOW)            │
│                                                    (t -> a, SUB)                 │
│                                                    (+ -> a, RIGHT)               │
│                                                    (b -> +, RIGHT)               │
│                                                                                  │
│  3D Tensor: [num_tokens x num_tokens x num_relations]                          │
│  Relations: RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE                              │
│                                                                                  │
│  Benefits:                                                                      │
│    • Smaller vocab (no _, ^, {, } syntax tokens)                               │
│    • Matches user input (user writes "t", not "_")                             │
│    • Structure explicit in relations                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Position Head Architecture: Tree-aware Position Module (TPM)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TPM: REUSES TRANSFORMER K CACHE + RELATION EMBEDDINGS                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Key Insight: h_t was produced by attending to transformer's K cache           │
│  → TPM should use the SAME K cache for consistency!                            │
│                                                                                  │
│  Query: concat(h_t, h_VC)  ← h_t for context, h_VC for spatial stroke location │
│  Keys:  K_transformer + relation_emb (additive conditioning)                   │
│                                                                                  │
│  Why reuse transformer's K cache?                                              │
│  ═══════════════════════════════════════════════════════════                   │
│  • h_t "knows" what it attended to during AR                                   │
│  • K cache contains positional encodings h_t learned from                      │
│  • No redundant key projection - just add relation embeddings                  │
│  • Full consistency: h_t's knowledge matches TPM's keys                        │
│                                                                                  │
│  Action Keys Construction:                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │  K_transformer[:, h, :, :]  +  relation_proj[h](rel_emb)        │           │
│  │       [B, N, d_head]              [R, d_head]                   │           │
│  │              ↓                                                   │           │
│  │       broadcast add → [B, N, R, d_head] → [B, N*R, d_head]      │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
│  Action Space (grows as we generate):                                          │
│  ┌────────────────────────────────────────┐                                    │
│  │ (frac, RIGHT)  (frac, SUP)  (frac, SUB)│                                    │
│  │ (a, RIGHT)     (a, SUP)     (a, SUB)   │ ← can attach to "a"                │
│  │ (c, RIGHT)     (c, SUP)     (c, SUB)   │                                    │
│  │ ... + newly generated tokens           │                                    │
│  └────────────────────────────────────────┘                                    │
│                                                                                  │
│  Multi-head scoring (head-to-head match with transformer):                     │
│    q_h = W_q_h(concat(h_t, h_VC))           → [d_head]                         │
│    k_h = K_cache[:, h] + relation_proj[h]   → [N*R, d_head]  (from cache!)     │
│    scores_h = q_h · k_h^T / √d_head         → [N*R]                            │
│    action_probs = softmax(avg_across_heads(scores))                            │
│                                                                                  │
│  Output: selected_action = argmax(action_probs)                                │
│          → (parent_idx, relation_type)                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### TPM vs SMM Design Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SMM vs TPM: UNIFIED QUERY, DIFFERENT KEY SOURCES                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MODULE │ QUERY             │ KEYS                    │ OUTPUT                 │
│  ═══════╪═══════════════════╪═════════════════════════╪════════════════════════│
│  SMM    │ concat(h_t, h_VC) │ Patch tokens (own proj) │ Stroke logits → sigmoid│
│  TPM    │ concat(h_t, h_VC) │ K_cache + relation_emb  │ Action logits → softmax│
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  UNIFIED QUERY: Both use concat(h_t, h_VC)                                     │
│  ═══════════════════════════════════════════                                    │
│    • h_t: AR hidden state (token identity + context structure)                 │
│    • h_VC: Visual Class token (spatial stroke positions)                       │
│    • Same query construction for architectural consistency                      │
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  SMM: "Which strokes contributed to THIS token?"                               │
│  ════════════════════════════════════════════════                               │
│    Keys: W_k(patch_tokens) — project patches to key space                      │
│    Scoring: q_h · k_patch → avg per stroke → sigmoid                           │
│    Output: Per-stroke binary selection [0,1]                                   │
│                                                                                  │
│  TPM: "Where does THIS token attach in the tree?"                              │
│  ═════════════════════════════════════════════════                              │
│    Keys: ROOT_key + relation_emb (for single-step training)                    │
│    Scoring: q_h · k_action → softmax                                           │
│    Output: Action probs over (parent, relation) pairs                          │
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  Why different key sources?                                                    │
│  ═══════════════════════════                                                    │
│    TPM: Predicting tree attachment → use expression structure (ROOT/K_cache)  │
│    SMM: Predicting stroke membership → use visual patch features              │
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  NOTE: [INT] token reserved for future teacher distillation                   │
│        Currently, h_t (from INT position) is used for all heads               │
│        Future: [INT] ← KL distillation from multimodal embedding teacher      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### AR Decoding with Dual Output

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EACH AR STEP = (SYMBOL, POSITION)                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Example: Adding "_t+b" to \frac{a}{c}                                         │
│                                                                                  │
│  Step │ Symbol │ Action        │ Meaning                │ LaTeX After          │
│  ═════╪════════╪═══════════════╪════════════════════════╪══════════════════════│
│   0   │   t    │ (a, SUB)      │ t subscript of a       │ \frac{a_t}{c}        │
│   1   │   +    │ (t, RIGHT)    │ + right of t           │ \frac{a_t+}{c}       │
│   2   │   b    │ (+, RIGHT)    │ b right of +           │ \frac{a_t+b}{c}      │
│   3   │ <EOS>  │   —           │ done                   │ \frac{a_t+b}{c}      │
│                                                                                  │
│  Note: No "_" token! SUB relation encodes subscript.                           │
│        Tree-to-LaTeX conversion adds syntax in post-processing.                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Option B: Incremental Stroke Removal During AR

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  WHY OPTION B? Model must see REMAINING strokes, not full image                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  WRONG (Option A): Same image every step                                        │
│  ════════════════════════════════════════                                       │
│  Step 1: See [a+b=c image], context=[], target="a"                             │
│  Step 2: See [a+b=c image], context=[a], target="+"  ← Still sees "a"!         │
│  Problem: Model can "cheat" - reads from image instead of learning             │
│                                                                                  │
│  CORRECT (Option B): Remove strokes after each step                            │
│  ══════════════════════════════════════════════════                            │
│  Step 1: See [a+b=c image], context=[], target="a"                             │
│          → SMM identifies strokes for "a" → REMOVE                             │
│  Step 2: See [+b=c image], context=[a], target="+"   ← "a" strokes gone!       │
│          → SMM identifies strokes for "+" → REMOVE                             │
│  Step 3: See [b=c image], context=[a,+], target="b"  ← "a+" strokes gone!      │
│                                                                                  │
│  Training: Teacher forcing with per-token stroke labels                        │
│  Inference: SMM predictions identify strokes to remove                         │
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  Data requirement: Per-token stroke labels                                      │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │  sample = {                                                      │           │
│  │      "tokens": ["a", "+", "b", "=", "c"],                       │           │
│  │      "stroke_labels": [                                          │           │
│  │          [0, 1],    # strokes 0,1 → "a"                         │           │
│  │          [2],       # stroke 2 → "+"                            │           │
│  │          [3, 4],    # strokes 3,4 → "b"                         │           │
│  │          [5],       # stroke 5 → "="                            │           │
│  │          [6, 7],    # strokes 6,7 → "c"                         │           │
│  │      ]                                                           │           │
│  │  }                                                               │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### K Cache Management with Recomputation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  K CACHE UPDATES: APPEND-ONLY vs RECOMPUTATION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Most actions: APPEND-ONLY (fast)                                               │
│  ════════════════════════════════                                               │
│  RIGHT, SUP, SUB: Linear tree growth, just append new token's key              │
│                                                                                  │
│  Step 1: K_cache = [K_a]                                                        │
│  Step 2: K_cache = [K_a, K_+]      ← append                                    │
│  Step 3: K_cache = [K_a, K_+, K_b] ← append                                    │
│                                                                                  │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                  │
│  Structural actions: RECOMPUTATION needed                                       │
│  ═════════════════════════════════════════                                      │
│  ABOVE, BELOW (fraction): Wraps existing tokens in new structure               │
│  INSIDE (sqrt, etc): Changes tree positions of wrapped tokens                  │
│                                                                                  │
│  Example: Writing "c" below "a+b" → \frac{a+b}{c}                              │
│                                                                                  │
│  Before: [K_a, K_+, K_b]        (linear)                                       │
│  After:  [K_frac, K_a', K_+', K_b', K_c]  (a,+,b now children of frac)        │
│          → Need to recompute K_a', K_+', K_b' if using tree positions          │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │  Relation        │ K Cache Action  │ Reason                     │           │
│  │  ════════════════╪═════════════════╪════════════════════════════│           │
│  │  RIGHT           │ Append          │ Linear sequence            │           │
│  │  SUP             │ Append          │ Parent unchanged           │           │
│  │  SUB             │ Append          │ Parent unchanged           │           │
│  │  ABOVE           │ RECOMPUTE       │ Creates fraction wrapper   │           │
│  │  BELOW           │ RECOMPUTE       │ Creates fraction wrapper   │           │
│  │  INSIDE          │ RECOMPUTE       │ Wraps in structure         │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Head B Detail: Score First, Then Pool

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MULTI-HEAD STROKE SELECTION (Score First, Then Pool)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DIMENSION DESIGN: Query has LESS compression (preserves context+visual)       │
│  ════════════════════════════════════════════════════════════════════          │
│                                                                                  │
│  d_concat = d_int + d_vc = 512 + 512 = 1024                                    │
│  d_patch = 512                                                                  │
│  num_heads = 4                                                                  │
│  d_head = d_concat / num_heads = 1024 / 4 = 256                                │
│                                                                                  │
│  Query: W_q_h(1024 → 256)  = 4× compression                                    │
│  Key:   W_k_h(512 → 256)   = 2× compression  ← LESS compressed                 │
│                                                                                  │
│  Why? Query carries richer signal (context + visual decision)                  │
│       Keys just need to be matchable, can afford more capacity                 │
│                                                                                  │
│  ────────────────────────────────────────────────────────────────────────────   │
│                                                                                  │
│  Patch tokens (grouped by stroke):                                              │
│  ════════════════════════════════                                               │
│  Stroke 1: [P₁₀, P₁₁, P₁₂]        (3 patches)                                  │
│  Stroke 2: [P₂₀, P₂₁]             (2 patches, incomplete)                      │
│                                                                                  │
│  Step 1: Project query and keys (asymmetric compression)                       │
│  ═══════════════════════════════════════════════════════                       │
│                                                                                  │
│  q_h = W_q_h(concat([INT],[VC]))   [1024] → [256]  (4× compression)           │
│  k_ij = W_k_h(P_ij)                [512]  → [256]  (2× compression)           │
│                                                                                  │
│  Step 2: Score EACH patch (per head)                                           │
│  ═══════════════════════════════════                                            │
│                                                                                  │
│  Head 1:  q₁·k₁₀=0.8  q₁·k₁₁=0.3  q₁·k₁₂=0.9  │  q₁·k₂₀=0.2  q₁·k₂₁=0.1     │
│  Head 2:  q₂·k₁₀=0.7  q₂·k₁₁=0.4  q₂·k₁₂=0.8  │  q₂·k₂₀=0.3  q₂·k₂₁=0.2     │
│  Head 3:  q₃·k₁₀=0.9  q₃·k₁₁=0.2  q₃·k₁₂=0.7  │  q₃·k₂₀=0.1  q₃·k₂₁=0.1     │
│  Head 4:  q₄·k₁₀=0.6  q₄·k₁₁=0.5  q₄·k₁₂=0.8  │  q₄·k₂₀=0.2  q₄·k₂₁=0.3     │
│                                                                                  │
│  Step 3: Average within each stroke (per head)                                  │
│  ═════════════════════════════════════════════                                  │
│                                                                                  │
│  Head 1:  stroke₁ = mean(0.8,0.3,0.9) = 0.67   stroke₂ = mean(0.2,0.1) = 0.15 │
│  Head 2:  stroke₁ = mean(0.7,0.4,0.8) = 0.63   stroke₂ = mean(0.3,0.2) = 0.25 │
│  Head 3:  stroke₁ = mean(0.9,0.2,0.7) = 0.60   stroke₂ = mean(0.1,0.1) = 0.10 │
│  Head 4:  stroke₁ = mean(0.6,0.5,0.8) = 0.63   stroke₂ = mean(0.2,0.3) = 0.25 │
│                                                                                  │
│  Step 4: Average across heads                                                   │
│  ════════════════════════════                                                   │
│                                                                                  │
│  final₁ = mean(0.67, 0.63, 0.60, 0.63) = 0.63  → σ(0.63) = 0.65 → commit ✓    │
│  final₂ = mean(0.15, 0.25, 0.10, 0.25) = 0.19  → σ(0.19) = 0.55 → keep        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Three-Head Training (with Option B)

| Component | Role | Training Signal |
|-----------|------|-----------------|
| **[VC]** | Visual Class (patches → visual embedding) | Implicit via SMM |
| **h_t** | AR hidden state (from INT position) | Used by all heads |
| **Head A** | Recognition: h_t → FFN → symbol | Cross-entropy |
| **SMM (B)** | Stroke: concat(h_t, h_VC) → patch scores | BCE on stroke labels |
| **TPM (C)** | Position: concat(h_t, h_VC) → ROOT actions | Cross-entropy on action |

**Note**: [INT] token reserved for future teacher distillation.

```
Loss = L_symbol + λ₁·L_position + λ₂·L_stroke_select + λ₃·KL([INT] ∥ Teacher)

Training Loop (Option B: Incremental Stroke Removal):
══════════════════════════════════════════════════════

for t in range(sequence_length):
    # 1. Compute on CURRENT image (previous strokes removed)
    masked_patches = remove_strokes(patches, already_removed)
    h_int, h_vc = compute_tokens(masked_patches, context)
    h_t, k_cache = transformer_step(masked_patches, context)
    
    # 2. Head A: Symbol prediction
    L_symbol += CE(symbol_head(h_t), target_tokens[t])
    
    # 3. TPM: Position prediction (uses K_cache!)
    q = concat(h_t, h_VC)
    action_keys = K_cache + relation_emb      # Reuse transformer's K!
    action_probs = softmax(q · action_keys)
    L_position += CE(action_probs, target_positions[t])
    
    # 4. SMM: Stroke selection
    stroke_scores = SMM(h_int, h_vc, masked_patches, active_strokes)
    L_stroke += BCE(stroke_scores, stroke_labels[t])
    
    # 5. REMOVE strokes for this token (teacher forcing)
    already_removed |= stroke_labels[t]

Key differences from naive approach:
  • Image CHANGES each step (strokes removed)
  • Model learns to recognize REMAINING strokes
  • SMM gets per-token stroke labels
  • TPM uses transformer's K_cache (not separate projection)
```

---

## 2. Minimum Experiments

| Exp | Question | Method | Success Metric |
|-----|----------|--------|----------------|
| **1** | Trajectory patches work? | Recognition accuracy | ExpRate > 70%, 10× fewer patches |
| **2** | [INT] captures reasoning? | t-SNE + classification | Clusters by problem type, acc > 80% |
| **3** | Efficiency gains real? | Measure ops/memory | O(n²) vs O(n³) demonstrated |

---

## 3. Data & Augmentation Strategy

```
MathWriting (ready):
├── synthetic/              396K expressions with strokes
├── synthetic-bboxes.jsonl  Per-symbol bounding boxes ← KEY
└── symbols/                6,423 individual symbol strokes
```

### Compositional Augmentation (Simulates Edit Scenarios)

```
1. COMPOSE: Concatenate 2-4 MathWriting chunks → context
   "x^2+1" + "\frac{a}{b}" + "=0" → "x^2+1, \frac{a}{b}, =0"
   
2. COMMIT: Context becomes TEXT tokens (not patches)
   
3. EDIT: Replace one chunk with different MathWriting sample
   • Context: "x^2+1, [MASK], =0" (text, KV-cached)
   • Edit chunk: render "\sqrt{n}" + stroke artifacts → image
   • Conv2D: image → 4×4 patch grid (standard, no trajectory tracking)
   • Output: "\sqrt{n}" [EOS]

This proves O(n²): context=text, only edit chunk=image patches
```

### Stroke Artifacts (Incomplete Strokes as Distractor)

```
Image = ONLY unrecognized strokes (snowballing until commit)

If complete chunks A and B both in image → model outputs "A, B" (both)
So artifacts must be INCOMPLETE strokes, not full chunks

Training setup:
  Image: [complete chunk strokes] + [1-2 incomplete strokes from NEXT symbol]
  Target: complete chunk only (ignore incomplete)

Source options:
  1. MathWriting: take first k strokes of another symbol (k < total)
  2. stroke_corpus/: random incomplete shreds

Model learns: complete → output, incomplete → wait/ignore
```

### Tree-to-LaTeX Conversion (Hardcoded Post-Processing)

```
MODEL OUTPUTS DURING AR:
════════════════════════
  • Content tokens ONLY: "a", "+", "b", "2", "x", etc.
  • Relations via TPM: RIGHT, SUP, SUB, ABOVE, BELOW, INSIDE

MODEL NEVER OUTPUTS:
════════════════════
  • LaTeX syntax: \frac, {, }, ^, _, \sqrt, etc.
  • These are added by hardcoded tree-to-LaTeX conversion!

CONVERSION RULES:
═════════════════
  RIGHT  → concatenation:     a RIGHT b  →  "ab"
  SUP    → superscript:       a SUP 2    →  "a^{2}"
  SUB    → subscript:         a SUB i    →  "a_{i}"
  ABOVE  → fraction num:      / ABOVE a  →  (builds \frac{a}{...})
  BELOW  → fraction denom:    / BELOW b  →  (builds \frac{...}{b})
  INSIDE → structure:         √ INSIDE x →  "\sqrt{x}"

EXAMPLE: User writes fraction
══════════════════════════════

  User writes:     a + b    (strokes at y=0-20)
                   ─────    (stroke at y=25-30) ← fraction bar
                     2      (stroke at y=35-50)

  Model AR output:
  ┌──────┬───────┬──────────────────┬────────────────┐
  │ Step │ Token │ Action           │ Tree           │
  ├──────┼───────┼──────────────────┼────────────────┤
  │  1   │  "a"  │ (ROOT, START)    │ a              │
  │  2   │  "+"  │ (a, RIGHT)       │ a → +          │
  │  3   │  "b"  │ (+, RIGHT)       │ a → + → b      │
  │  4   │  "2"  │ (b, BELOW)       │ a → + → b      │
  │      │       │                  │          ↓     │
  │      │       │                  │          2     │
  └──────┴───────┴──────────────────┴────────────────┘

  Hardcoded conversion:
    Tree with BELOW relation → detect fraction pattern
    → Wrap (a, +, b) as numerator, (2) as denominator
    → Output: "\frac{a+b}{2}"

IMPLEMENTATION:
═══════════════
  context_tracker/model/tree_to_latex.py:
    • ExpressionTree: builds tree from tokens + actions
    • TreeToLatex: converts tree to LaTeX string
    • SpatialRelationInferrer: infers relations from bboxes
```

### Spatial Relation Inference (Training Label Generation)

```
FROM BBOXES TO ACTION LABELS:
═════════════════════════════

  MathWriting provides per-symbol bounding boxes:
    symbol="a", bbox=(0, 10, 15, 30)   ← x at y=10-30
    symbol="2", bbox=(18, 5, 25, 15)   ← 2 higher, to right → SUP

  SpatialRelationInferrer detects:
  ┌─────────────────────────────────────────────────────────────┐
  │ Condition                          │ Inferred Relation      │
  ├─────────────────────────────────────────────────────────────┤
  │ Child above parent, large overlap  │ ABOVE (fraction num)   │
  │ Child below parent, large overlap  │ BELOW (fraction denom) │
  │ Child above parent, to right       │ SUP (superscript)      │
  │ Child below parent, to right       │ SUB (subscript)        │
  │ Child to right, similar height     │ RIGHT (sequence)       │
  │ Child overlaps parent interior     │ INSIDE (sqrt/matrix)   │
  └─────────────────────────────────────────────────────────────┘

  Used in: context_tracker/training/dataset.py
    → Generates proper target_actions for TPM training
    → Enables fraction, subscript, superscript learning
```

---

## 4. Implementation Checklist

### Core Modules (Completed ✓)
- [x] **SMM**: Multi-head stroke modification module (`module/stroke_modification.py`)
- [x] **TPM**: Tree-aware position module with K-cache reuse (`module/position_head.py`)
- [x] **KCacheManager**: K-cache with append/recomputation logic (`module/k_cache_manager.py`)
- [x] **VisualPatchEncoder**: Stacked 3x3 conv → 8×8 patches (`module/visual_encoder.py`)
- [x] **RelationshipTensor**: LaTeX → 3D relation tensor (`module/relationship_tensor.py`)
- [x] **ContextTrackerModel**: Full model integration (`context_tracker.py`)
- [x] **Tree-to-LaTeX**: Hardcoded conversion (`tree_to_latex.py`)
- [x] **SpatialRelationInferrer**: Bbox → action labels (`tree_to_latex.py`)

### Data Pipeline (Completed ✓)
- [x] MathWriting atomic loader with per-symbol bboxes (`data/mathwriting_atomic.py`)
- [x] ChunkRenderer: strokes → image + artifact overlay
- [x] CompositionalAugmentor: context + edit chunk generation
- [x] EditDataset with proper action label inference (`training/dataset.py`)

### Training (Pending)
- [ ] Train recognition: context (text) + chunk (image) → LaTeX
- [ ] Add [INT] token with teacher distillation
- [ ] Extract trajectory embeddings from [INT]
- [ ] t-SNE visualization + classification

### Paper
- [ ] Efficiency comparison table
- [ ] Figures: architecture, t-SNE, efficiency
- [ ] Write method section

---

## 5. Key Figures for Paper

| Figure | Content | Priority |
|--------|---------|----------|
| Fig 1 | Architecture (dual-token, teacher distillation) | Must |
| Fig 2 | **t-SNE of [INT] embeddings** (clusters by problem type) | ⭐ KEY |
| Fig 3 | Efficiency: patches, memory, O(n²) vs O(n³) | Must |
| Fig 4 | SMM and TPM module detail (multi-head attention flow) | Must |
| Table 1 | Quantitative: ExpRate, classification acc, speedup | Must |

### TikZ Diagrams (Publication-Ready)

```
context_tracker/docs/
├── architecture_diagram.pdf   # Overview: inputs → transformer → 3 heads
└── smm_tpm_detail.pdf         # Internal SMM + TPM: query/key projections, scoring
```

Compile with: `pdflatex architecture_diagram.tex`

---

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| Expression Recognition Rate | > 70% |
| Trajectory Classification | > 80% |
| Patch Reduction | 5-10× fewer than dense |
| Cumulative Cost | O(n²) proven |

---

*Target: 2-3 week prototype*
