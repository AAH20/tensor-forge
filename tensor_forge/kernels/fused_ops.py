"""
Fused Single-Pass Kernel Implementations.
Combines memory-bandwidth bound operations (e.g. RMSNorm + SwiGLU) into a single SRAM-resident pass.
"""
import math
from typing import List

class FusedKernelEngine:
    @staticmethod
    def fused_rmsnorm_swiglu(
        x: List[float],
        gate_weights: List[float],
        up_weights: List[float],
        norm_weight: float = 1.0,
        eps: float = 1e-6
    ) -> List[float]:
        """
        Single-pass execution:
        1. Calculate RMS of x
        2. Normalize x in SRAM cache
        3. Fused SwiGLU: (norm_x * gate) * sigmoid(norm_x * gate) * (norm_x * up)
        Never writes intermediate normalized activations back to DRAM.
        """
        n = len(x)
        if n == 0:
            return []

        # Step 1: Compute Root Mean Square in fast accumulator
        sq_sum = sum(v * v for v in x)
        rms = math.sqrt(sq_sum / n + eps)
        inv_rms = 1.0 / rms

        # Step 2: Fused Element-wise Projection & SwiGLU
        out = []
        for i in range(n):
            norm_val = x[i] * inv_rms * norm_weight
            gate = norm_val * gate_weights[i]
            up = norm_val * up_weights[i]
            # SiLU / Swish(gate) = gate / (1 + exp(-gate))
            silu_gate = gate / (1.0 + math.exp(-max(-80.0, min(80.0, gate))))
            out.append(silu_gate * up)

        return out
