import torch
import torch.nn as nn
import torch.nn.functional as F

#  restimate
class PruneRatioEMA:
    def __init__(self, beta=0.80, r_min=0.05, r_max=0.4, eps=1e-8):
        self.beta = beta
        self.r_min = r_min
        self.r_max = r_max
        self.eps = eps

    @torch.no_grad()
    def batch_prune_ratio(self, x_a, x_b, y):
        # mean-pool
        a = F.normalize(x_a.mean(dim=1), dim=-1, eps=self.eps)   # [B, D]
        b = F.normalize(x_b.mean(dim=1), dim=-1, eps=self.eps)   # [B, D]
        yb = F.normalize(y.mean(dim=1), dim=-1, eps=self.eps)    # [B, D]

        c_a = (a * yb).sum(dim=-1).abs()                         # [B]
        c_b = (b * yb).sum(dim=-1).abs()                         # [B]
        S_a = (c_a / (c_a + c_b + self.eps)).mean()              # scalar

        keep = self.r_min + (self.r_max - self.r_min) * S_a
        return 1.0 - keep












