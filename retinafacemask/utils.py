from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class PolyLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group according to poly learning rate policy
    """

    def __init__(self, optimizer: Optimizer, max_iter: int = 90000, power: float = 0.9, last_epoch: int = -1) -> None:
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1 - float(self.last_epoch) / self.max_iter) ** self.power
            for base_lr in self.base_lrs
        ]
