
import numpy as np

class FrequencyEstimator:
    """Streaming frequency estimator for item sampling probability p_j (Algorithm 2)."""
    def __init__(self, H=5_000_000, alpha=0.01, init_gap=100.0):
        self.H = int(H)
        self.alpha = float(alpha)
        self.A = np.zeros(self.H, dtype=np.int64)
        self.B = np.full(self.H, float(init_gap), dtype=np.float64)

    def _h(self, y):
        return (hash(int(y)) if isinstance(y, (int, np.integer)) else hash(str(y))) % self.H

    def update_batch(self, ids, t):
        for y in ids:
            h = self._h(y)
            gap = t - self.A[h]
            self.B[h] = (1.0 - self.alpha) * self.B[h] + self.alpha * float(gap)
            self.A[h] = t

    def get_probs(self, ids):
        out = np.empty(len(ids), dtype=np.float64)
        for i, y in enumerate(ids):
            h = self._h(y)
            gap = max(self.B[h], 1e-6)
            out[i] = 1.0 / gap
        return out
