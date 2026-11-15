from torch.optim.lr_scheduler import LambdaLR


class _SchedulerConfigMixin:
    """Exposes LRSchedulerConfig fields as constructor args so they can be set in YAML."""
    def _set_scheduler_config(self, interval: str = "epoch", frequency: int = 1):
        assert interval in ("step", "epoch")
        self.interval = interval
        self.frequency = frequency

class WarmupStepLR(LambdaLR, _SchedulerConfigMixin):
    """linear warmup over `warmup` steps, then decay by
    `factor` every `patience` steps, floored at `min_lr` (absolute)."""
    def __init__(self, optimizer, warmup: int, patience: int, factor: float,
                 min_lr: float = 0.0, interval: str = "step", frequency: int = 1):
        base_lr = optimizer.param_groups[0]["lr"]
        floor = min_lr / base_lr if base_lr > 0 else 0.0
        def lr_lambda(n):                      # n = scheduler.step() count
            if n < warmup:
                return max((n + 1) / warmup, floor)
            return max(factor ** ((n - warmup) // patience), floor)
        super().__init__(optimizer, lr_lambda)
        self._set_scheduler_config(interval, frequency)