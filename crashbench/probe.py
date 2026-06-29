"""Self-report probe — load & score (Path 1-1a, PLAN.md §11 Stage 1).

The probe is the SAME linear crash-imminence decoder trained in
`scripts/probe_selfreport_analysis.py` (PCA-50 + logistic regression on OpenVLA's
frozen last hidden state, LOSO AUC 0.99-1.0). That script dumps its T=5 parameters
to `results/selfreport/probe_T5.npz`; this module reloads them so a live policy can
score a single hidden state h (4096d) on every step:

    z     = (h - mu_pca) @ V_pca            # PCA project to 50d
    logit = [(z - mu_lr) / sd_lr, 1] . w_lr # standardize + bias, linear decoder

`logit > thr` => "I am about to crash". The threshold was fixed on the negative
pool (off-path + no-wall frames) so the guard (almost) never fires absent a real
on-path crash. For activation steering (Path 1-1b) `steer_vector()` lifts the
decoder direction back to the 4096d residual stream.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PATH = "results/selfreport/probe_T5.npz"


class Probe:
    def __init__(self, mu_pca, V_pca, mu_lr, sd_lr, w_lr, thr):
        self.mu_pca = np.asarray(mu_pca, dtype=np.float32)   # (4096,)
        self.V_pca = np.asarray(V_pca, dtype=np.float32)     # (4096, k)
        self.mu_lr = np.asarray(mu_lr, dtype=np.float32)     # (k,)
        self.sd_lr = np.asarray(sd_lr, dtype=np.float32)     # (k,)
        self.w_lr = np.asarray(w_lr, dtype=np.float32)       # (k+1,) last entry = bias
        self.thr = float(thr)

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Probe":
        d = np.load(path)
        return cls(d["mu_pca"], d["V_pca"], d["mu_lr"], d["sd_lr"], d["w_lr"], float(d["thr"]))

    def logit(self, h) -> float:
        """Crash-imminence logit for one hidden state h (4096d). Higher => more imminent."""
        h = np.asarray(h, dtype=np.float32).reshape(-1)
        z = (h - self.mu_pca) @ self.V_pca           # (k,)
        zs = (z - self.mu_lr) / self.sd_lr
        return float(np.dot(zs, self.w_lr[:-1]) + self.w_lr[-1])

    def fires(self, h) -> bool:
        return self.logit(h) > self.thr

    def steer_vector(self) -> np.ndarray:
        """Unit 4096d direction in the residual stream that INCREASES the crash logit.
        Subtracting alpha*this from the hidden state pushes the model away from
        'I will crash' (Path 1-1b activation steering). Derived by mapping the decoder
        weight (in standardized PCA coords) back through V_pca:
            d_4096 = V_pca @ (w_lr[:-1] / sd_lr)
        """
        d = self.V_pca @ (self.w_lr[:-1] / self.sd_lr)
        n = np.linalg.norm(d)
        return (d / n).astype(np.float32) if n > 0 else d.astype(np.float32)
