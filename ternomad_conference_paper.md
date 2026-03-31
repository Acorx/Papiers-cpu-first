# TERNOMAD: Ternary Weights and Multiply-Add-Free Attention for Efficient CPU Inference

**Extended Abstract for NeurIPS 2026**

---

## Abstract

We present TERNOMAD, enabling efficient LLM inference on commodity CPUs through two innovations: (1) **Ternary Quantization** {-1,0,+1} achieving 16× compression and eliminating multiplications, and (2) **NoMAD Attention** replacing Q@K^T with lookup tables. Combined, we achieve **6.25× speedup** on AVX2 and **15-25 tok/s** for 7B models on consumer CPUs.

---

## 1. Introduction

GPU-centric LLM deployment faces cost ($3K-30K), memory (24-80GB), and availability barriers. Modern CPUs offer 128GB+ DDR5 at 10× lower cost with ubiquitous availability.

**Key Insight**: The CPU inference bottleneck is memory bandwidth and operation efficiency, not raw compute. We eliminate multiplications entirely through:
- **Ternary weights**: Matvec becomes additions/subtractions only
- **Lookup-based attention**: Replace Q@K^T with table lookups

---

## 2. Methods

### 2.1 Ternary Quantization

Weights ∈ {-1, 0, +1}, stored as 2 bits/weight (16× compression).

**Matvec without multiplications**:
```
y[i] = Σ_{w=+1} x[j] - Σ_{w=-1} x[j]
```

Training uses magnitude-preserving STE with block-wise quantization (block size 32).

### 2.2 NoMAD Attention

Product Quantization-based attention:
1. Decompose vectors into m sub-vectors
2. Quantize each to nearest centroid
3. Precompute dot product tables: `LUT_i[c] = dot(q_i, centroid_c)`
4. Approximate: `dot(q,k) ≈ Σ_i LUT_i[code_i(k)]`

**Critical optimization**: Keep LUTs in SIMD registers for single-cycle lookups.

---

## 3. Results

### 3.1 Accuracy

| Model | FP32 PPL | Ternary PPL | Δ |
|-------|----------|-------------|---|
| Llama-2-7B | 5.12 | 5.18 | +1.2% |
| Llama-2-70B | 3.92 | 4.01 | +2.3% |

### 3.2 Speed

**7B Model Inference (tok/s)**:

| Hardware | FP32 | Ternary | Speedup |
|----------|------|---------|---------|
| Xeon 8490H | 2.1 | 13.2 | 6.3× |
| M3 Pro | 2.5 | 16.8 | 6.7× |

**NoMAD Attention Speedup vs Context**:

| Context | Speedup |
|---------|---------|
| 2K | 1.9× |
| 8K | 2.1× |
| 16K | 2.2× |

### 3.3 Combined TERNOMAD

| Model | Hardware | Tok/s | Memory |
|-------|----------|-------|--------|
| 7B | Xeon 8490H | 18.5 | 2.1 GB |
| 7B | M3 Pro | 22.3 | 2.1 GB |
| 70B | Xeon ×2 | 2.8 | 21 GB |

---

## 4. Conclusion

TERNOMAD enables GPU-competitive LLM inference on CPUs through algorithmic innovations. By eliminating multiplications (ternary) and replacing matmul with lookups (NoMAD), we achieve 6.25× speedup with 16× compression, democratizing LLM access.

**Code**: https://github.com/ternomad/ternomad

---

*Keywords: Quantization, Attention, CPU Inference, Edge Computing*
