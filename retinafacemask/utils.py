import re
from pathlib import Path
from typing import Union, Optional, Any, Dict, Tuple

import numpy as np
import torch
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
        return [base_lr * (1 - float(self.last_epoch) / self.max_iter) ** self.power for base_lr in self.base_lrs]


def load_checkpoint(file_path: Union[Path, str], rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """Loads PyTorch checkpoint, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        checkpoint["state_dict"] = result

    return checkpoint


def random_color() -> Tuple[int, ...]:
    result = tuple(np.random.randint(0, 255, size=3))
    return tuple([int(x) for x in result])
