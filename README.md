# TensorForge: Bare-Metal Deep-Kernel Fused JIT & Memory Bandwidth Optimizer

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Hardware: Tenstorrent / CUDA / Apple Silicon](https://img.shields.io/badge/Hardware-Tenstorrent%20%7C%20CUDA%20%7C%20Apple%20Silicon-purple.svg)](https://a2zsoc.com)
[![Benchmarks: Exceeding Eager PyTorch](https://img.shields.io/badge/Benchmarks-4.5x%20DRAM%20Savings-orange.svg)](https://a2zsoc.com)

> **Eliminating Memory-Bandwidth Bottlenecks and Numerical Overflows in Frontier AI Accelerators.**  
> Automated graph pattern fusion, zero-overflow elementary reformulations, and strict double-sided saturation quantization for Tenstorrent, Triton, CUDA, and Apple Silicon MLX.

---

## 🎯 The Problem

Frontier AI inference models (Llama 3, DeepSeek, Mistral) are heavily **memory-bandwidth bound**. In eager PyTorch or naive compiler backends:
1. **DRAM Round-Trips**: Chains of element-wise operators (`RMSNorm -> SiLU -> Mul`) repeatedly read and write intermediate activations back to high-latency DRAM.
2. **Numerical Instability / Saturation Flaws**:
   - `logaddexp` and `logaddexp2` naive implementations evaluate $\log(\exp(a) + \exp(b))$, overflowing to $\pm\infty$ at $|x| > 88.7$.
   - Quantization kernels for `uint8` often lack explicit lower-bound saturation, resulting in magnitude reflection bugs where negative numbers wrap to positive bytes.

---

## ⚡ TensorForge Solution & Industry Benchmarks

| Feature | Naive Eager Baseline (PyTorch / Standard LLKs) | **TensorForge Fused JIT** | Performance / Safety Gain |
| :--- | :---: | :---: | :---: |
| **`logaddexp` Stability** | Overflows to `inf` at $|x| > 88.7$ | **Zero overflow across $(-\infty, +\infty)$** | Mathematically stable via log-sum-exp reformulation |
| **`uint8` Quantization Saturation** | Magnitude reflection on negative values | **Strict $[0, 255]$ double-sided clamping** | Eliminates activation corruption in INT8/FP8 models |
| **DRAM Memory Traffic** | 9 round-trip tensor transfers (576 MB) | **Single-pass SRAM fusion (128 MB)** | **4.5x DRAM traffic reduction (77.8% bandwidth saved)** |
| **Roofline Optimization** | Sub-optimal DRAM thrashing | **Automated Roofline regime targeting** | Maximum achievable arithmetic intensity |

---

## 🛠️ Architecture

```
tensor-forge/
├── tensor_forge/
│   ├── kernels/
│   │   ├── overflow_safe.py     # Overflow-safe logaddexp, logaddexp2, and stable softmax
│   │   ├── quant_saturation.py  # Double-sided saturated UINT8/INT8 quantize/requantize
│   │   └── fused_ops.py         # Single-pass SRAM-resident RMSNorm + SwiGLU engine
│   ├── compiler/
│   │   ├── graph_fusion.py      # Subgraph pattern matcher and operator fusion optimizer
│   │   └── memory_planner.py    # Zero-allocation static SRAM scratchpad planner
│   └── benchmarks/
│       ├── roofline.py          # Roofline model analyzer (H100, M3 Max, Blackhole)
│       └── memory_traffic.py    # DRAM vs SRAM byte reduction benchmark
```

---

## 💻 Quick Start & CLI

```bash
# Run unit tests
python3 -m unittest discover -s tests

# 1. Verify zero-overflow elementary operations
tensor-forge verify-overflow

# 2. Verify double-sided uint8 lower-bound saturation
tensor-forge verify-quant

# 3. Analyze Roofline Model on Tenstorrent Blackhole
tensor-forge roofline --hw tenstorrent_blackhole --flops 50000000 --bytes 1000000

# 4. Benchmark DRAM memory traffic reduction
tensor-forge memory-traffic --seq 4096 --dim 8192
```

---

## 📄 License & Enterprise Retainers

Apache-2.0 License. Authored by [Ahmed Hassan](https://github.com/AAH20) (Founder, [A2Z SOC](https://a2zsoc.com)).  
For bespoke kernel fusion, hardware accelerator SDKs, and inference optimization retainers, contact: `ahmed@a2zsoc.com`.
