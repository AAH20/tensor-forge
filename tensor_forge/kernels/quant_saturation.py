"""
Strict Lower/Upper Bound Saturated Quantization.
Eliminates magnitude reflection bugs (where -x produced identical byte outputs to x)
for UINT8 and INT8 quantization kernels.
"""
from typing import List

class QuantizationSaturation:
    @staticmethod
    def quantize_uint8(values: List[float], scale: float, zero_point: int) -> List[int]:
        """
        output = clamp(round(input / scale + zero_point), 0, 255)
        Enforces strict lower-bound saturation at 0 (never wrapping or magnitude-reflecting).
        """
        output = []
        for x in values:
            q = round(x / scale + zero_point)
            clamped = max(0, min(255, q))
            output.append(int(clamped))
        return output

    @staticmethod
    def requantize_uint8(
        uint8_inputs: List[int],
        input_scale: float,
        input_zero_point: int,
        output_scale: float,
        output_zero_point: int
    ) -> List[int]:
        """
        Requantization with double-sided saturation:
        output = clamp(round((input - in_zp) * in_scale / out_scale + out_zp), 0, 255)
        """
        output = []
        for q in uint8_inputs:
            real_val = (q - input_zero_point) * input_scale
            new_q = round(real_val / output_scale + output_zero_point)
            clamped = max(0, min(255, new_q))
            output.append(int(clamped))
        return output
