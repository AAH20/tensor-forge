"""
Overflow-Safe Elementary Functions.
Resolves numerical blowups and ±inf errors in log-sum-exp, logaddexp, and softmax operations
outside IEEE-754 standard activation ranges.
"""
import math
from typing import List, Tuple

class OverflowSafeOps:
    @staticmethod
    def logaddexp(a: float, b: float) -> float:
        """
        Calculates log(exp(a) + exp(b)) without overflow:
        logaddexp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
        """
        if math.isinf(a) and a < 0:
            return b
        if math.isinf(b) and b < 0:
            return a
        if a == b:
            return a + math.log(2.0)
        
        max_val = max(a, b)
        diff = -abs(a - b)
        return max_val + math.log1p(math.exp(diff))

    @staticmethod
    def logaddexp2(a: float, b: float) -> float:
        """
        Base-2 logaddexp: log2(2^a + 2^b)
        logaddexp2(a, b) = max(a, b) + log2(1 + 2^(-abs(a - b)))
        """
        if math.isinf(a) and a < 0:
            return b
        if math.isinf(b) and b < 0:
            return a
        if a == b:
            return a + 1.0

        max_val = max(a, b)
        diff = -abs(a - b)
        return max_val + math.log2(1.0 + math.pow(2.0, diff))

    @staticmethod
    def stable_softmax(logits: List[float]) -> List[float]:
        """
        Numerically stable softmax: subtract max before exponentiation.
        """
        if not logits:
            return []
        max_l = max(logits)
        exps = [math.exp(x - max_l) for x in logits]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]
