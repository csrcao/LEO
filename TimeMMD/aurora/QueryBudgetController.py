import torch

# coladapt
class AdaptiveBudgetController:
    def __init__(
        self,
        target_valid_ratio=0.105,
        K_min=16,
        K_max=None,
        k_init=None,
        ratio_init=0.05,
        lr=0.3,
        use_feedback=True,
    ):
        self.target = target_valid_ratio
        self.K_min = K_min
        self.K_max = K_max
        self.k = k_init
        self.query_ratio = ratio_init
        self.lr = lr
        self.use_feedback = use_feedback
        self._last_valid_ratio = None

    @torch.no_grad()
    def compute_K(self, k_from_outside):
        if self.k is None:
            self.k = int(k_from_outside)
            self.K_max = int(1.5*k_from_outside)

        return self.k

    def get_ratio(self):
        return self.query_ratio

    @torch.no_grad()
    def update(self, valid_ratio: float):
        if not self.use_feedback:
            return self.query_ratio

        self._last_valid_ratio = valid_ratio
        error = valid_ratio - self.target

        self.query_ratio *= torch.exp(torch.tensor(self.lr * error)).item()
        self.query_ratio = float(max(1e-4, min(10.0, self.query_ratio)))

        adjust = int(0.5 * error * self.k)
        self.k = int(self.k - adjust)
        self.k = max(self.K_min, min(self.K_max, self.k))

        return self.query_ratio


