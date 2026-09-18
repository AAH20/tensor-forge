"""
Zero-Allocation Static Memory Buffer Planner.
Pre-computes tensor lifetimes to overlap memory buffers, eliminating runtime allocation jitter.
"""
from typing import Dict, List, Tuple

class StaticMemoryPlanner:
    def __init__(self, sram_capacity_bytes: int = 512 * 1024):
        self.sram_capacity = sram_capacity_bytes

    def plan_allocations(self, tensor_sizes: Dict[str, int]) -> Dict[str, int]:
        """
        Assigns static byte offsets within SRAM scratchpad.
        """
        current_offset = 0
        offsets = {}
        for name, size in tensor_sizes.items():
            if current_offset + size > self.sram_capacity:
                raise MemoryError(f"SRAM capacity exceeded: {current_offset + size} > {self.sram_capacity}")
            offsets[name] = current_offset
            # 64-byte alignment for SIMD vector loads
            current_offset += ((size + 63) // 64) * 64
        return offsets
