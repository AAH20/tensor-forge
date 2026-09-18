"""
DRAM vs SRAM Memory Traffic Reduction Estimator.
Quantifies exact round-trip byte savings achieved by operator fusion.
"""
from typing import Dict, Any

class MemoryTrafficBenchmark:
    @staticmethod
    def compare_unfused_vs_fused(sequence_length: int, hidden_dim: int, bytes_per_elem: int = 2) -> Dict[str, Any]:
        # Unfused RMSNorm -> SwiGLU:
        # RMSNorm: reads X, writes NormX (2 passes)
        # Gate Linear: reads NormX, writes Gate (2 passes)
        # Up Linear: reads NormX, writes Up (2 passes)
        # SiLU & Mul: reads Gate & Up, writes Out (3 passes)
        # Total Unfused Memory Traffic: ~9 tensor DRAM round-trips
        tensor_bytes = sequence_length * hidden_dim * bytes_per_elem
        unfused_traffic_bytes = tensor_bytes * 9

        # Fused RMSNorm + SwiGLU:
        # Single pass reads X, computes everything in SRAM registers, writes Out
        # Total Fused Memory Traffic: reads X + writes Out = 2 tensor DRAM round-trips
        fused_traffic_bytes = tensor_bytes * 2

        bytes_saved = unfused_traffic_bytes - fused_traffic_bytes
        speedup_factor = round(float(unfused_traffic_bytes) / float(fused_traffic_bytes), 2)

        return {
            "tensor_size_bytes": tensor_bytes,
            "unfused_traffic_bytes": unfused_traffic_bytes,
            "fused_traffic_bytes": fused_traffic_bytes,
            "bytes_saved": bytes_saved,
            "memory_traffic_reduction_ratio": f"{speedup_factor}x",
            "bandwidth_efficiency_gain_pct": round((1.0 - fused_traffic_bytes / unfused_traffic_bytes) * 100, 1)
        }
