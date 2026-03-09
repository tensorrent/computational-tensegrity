"""Sigma-Engine: Spectral Proximity Instrument. SIP License v1.1."""
import numpy as np
from scipy.linalg import solve_lyapunov
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SigmaResult:
    rho: float; rho_dot: float; basin_proxy: float
    escape_prob: float; instability_risk: float

class SigmaEngine:
    def __init__(self, window=20, delta=10.0):
        self.window = window; self.delta = delta
        self.history = []; self.rho_history = []
    def estimate_jacobian(self, stream):
        if len(stream) < self.window: return np.eye(len(stream[0]))
        X, Y = np.array(stream[:-1]).T, np.array(stream[1:]).T
        J, *_ = np.linalg.lstsq(X.T, Y.T, rcond=None)
        return J.T
    def white_noise_probe(self, J, eps=0.05, trials=50, steps=200):
        n, escapes = J.shape[0], 0
        for _ in range(trials):
            x = np.zeros(n)
            for _ in range(steps):
                x = J @ x + eps * np.random.randn(n)
                if np.linalg.norm(x) > self.delta: escapes += 1; break
        p = escapes / trials
        return p, eps**2 * np.log(1/max(p, 1e-10))
    def process(self, x_new):
        self.history.append(x_new)
        if len(self.history) > self.window: self.history.pop(0)
        if len(self.history) >= self.window:
            J = self.estimate_jacobian(self.history)
            rho = float(np.max(np.abs(np.linalg.eigvals(J))))
            rho_dot = rho - self.rho_history[-1] if self.rho_history else 0
            self.rho_history.append(rho)
            p_esc, basin = self.white_noise_probe(J)
            osm = (1 - min(rho, 0.999)) * basin
            risk = 1 / (1 + osm)
            return SigmaResult(rho, rho_dot, basin, p_esc, risk)
        return None
