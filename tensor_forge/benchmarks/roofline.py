"""
Hardware Roofline Model Estimator.
Quantifies whether a kernel is Memory-Bound or Compute-Bound on target architectures
(NVIDIA H100, Apple M3 Max, Tenstorrent Wormhole/Blackhole).
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class HardwareProfile:
    name: str
    peak_tflops: float
    peak_bandwidth_gbs: float  # GB/s

HARDWARE_CATALOG = {
    "nvidia_h100": HardwareProfile("NVIDIA H100 SXM", 989.0, 3350.0),
    "apple_m3_max": HardwareProfile("Apple M3 Max GPU", 49.0, 400.0),
    "tenstorrent_blackhole": HardwareProfile("Tenstorrent Blackhole", 250.0, 1024.0)
}

class RooflineModelBenchmark:
    @staticmethod
    def analyze_kernel(profile_name: str, total_flops: int, memory_bytes_transferred: int) -> Dict[str, Any]:
        hw = HARDWARE_CATALOG.get(profile_name, HARDWARE_CATALOG["nvidia_h100"])
        # Operational Intensity (FLOPs / Byte)
        op_intensity = total_flops / float(max(1, memory_bytes_transferred))
        # Knee point (Balance point): Peak TFLOPs * 10^12 / (Peak Bandwidth * 10^9) = Peak TFLOPs * 1000 / Peak Bandwidth
        knee_point = (hw.peak_tflops * 1000.0) / hw.peak_bandwidth_gbs

        is_memory_bound = op_intensity < knee_point
        max_achievable_tflops = min(hw.peak_tflops, op_intensity * hw.peak_bandwidth_gbs / 1000.0)

        return {
            "hardware": hw.name,
            "operational_intensity_flops_per_byte": round(op_intensity, 3),
            "hardware_knee_point": round(knee_point, 2),
            "regime": "MEMORY_BOUND" if is_memory_bound else "COMPUTE_BOUND",
            "max_achievable_tflops": round(max_achievable_tflops, 2),
            "efficiency_vs_peak_pct": round((max_achievable_tflops / hw.peak_tflops) * 100, 1)
        }
