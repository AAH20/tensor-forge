import unittest
import math
from tensor_forge.kernels.overflow_safe import OverflowSafeOps
from tensor_forge.kernels.quant_saturation import QuantizationSaturation
from tensor_forge.kernels.fused_ops import FusedKernelEngine
from tensor_forge.compiler.graph_fusion import GraphFusionOptimizer, Node
from tensor_forge.benchmarks.roofline import RooflineModelBenchmark

class TestTensorForge(unittest.TestCase):
    def test_logaddexp_stability(self):
        # Extreme positive
        res_pos = OverflowSafeOps.logaddexp(100.0, 0.0)
        self.assertAlmostEqual(res_pos, 100.0, places=4)

        # Extreme negative
        res_neg = OverflowSafeOps.logaddexp(-100.0, -100.0)
        self.assertAlmostEqual(res_neg, -100.0 + math.log(2.0), places=4)

        # Base-2 logaddexp
        res_base2 = OverflowSafeOps.logaddexp2(200.0, 200.0)
        self.assertEqual(res_base2, 201.0)

    def test_quantization_lower_bound_saturation(self):
        inputs = [-100.0, -5.0, -0.01, 0.0, 128.0, 300.0]
        q = QuantizationSaturation.quantize_uint8(inputs, scale=1.0, zero_point=0)
        self.assertEqual(q[0], 0)
        self.assertEqual(q[1], 0)
        self.assertEqual(q[2], 0)
        self.assertEqual(q[3], 0)
        self.assertEqual(q[4], 128)
        self.assertEqual(q[5], 255)

    def test_fused_rmsnorm_swiglu(self):
        x = [1.0, 2.0, 3.0, 4.0]
        gate = [0.5, 0.5, 0.5, 0.5]
        up = [1.0, 1.0, 1.0, 1.0]
        out = FusedKernelEngine.fused_rmsnorm_swiglu(x, gate, up)
        self.assertEqual(len(out), 4)
        for v in out:
            self.assertTrue(v >= 0.0)

    def test_graph_fusion_pattern_matching(self):
        nodes = [
            Node("n1", "RMSNorm", ["x"], ["norm_x"]),
            Node("n2", "SiLU", ["norm_x"], ["act_x"]),
            Node("n3", "Mul", ["act_x"], ["out"]),
            Node("n4", "Linear", ["out"], ["final"])
        ]
        opt = GraphFusionOptimizer()
        fused = opt.optimize_graph(nodes)
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0]["op_type"], "FusedRMSNormSwiGLU")
        self.assertEqual(fused[1]["op_type"], "Linear")

    def test_roofline_analysis(self):
        analysis = RooflineModelBenchmark.analyze_kernel("tenstorrent_blackhole", 50_000_000, 1_000_000)
        self.assertEqual(analysis["regime"], "MEMORY_BOUND")
        self.assertTrue(analysis["max_achievable_tflops"] > 0)

if __name__ == "__main__":
    unittest.main()
