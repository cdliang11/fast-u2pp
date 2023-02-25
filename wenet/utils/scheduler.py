

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler

from typeguard import check_argument_types
import math

class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        assert check_argument_types()
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step


class DistillWeightLR:
    def __init__(self,
                 model,
                 epoch_iter,
                 increase_start_epoch,
                 fix_start_epoch,
                 initial_weight,
                 final_weight,
                 update_weight,
                 increase_type='exp'):
        self.increase_start_iter = (increase_start_epoch - 1) * epoch_iter
        self.fix_start_iter = (fix_start_epoch - 1) * epoch_iter
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.update_weight = update_weight

        self.fix_already = False
        self.current_iter = 0
        self.increase_type = increase_type
        self.model = model

        self.increase_iter = self.fix_start_iter - self.increase_start_iter
        self.init_weight()

    def init_weight(self):
        if hasattr(self.model, 'update_distill_weight'):
            self.model.update_distill_weight(self.initial_weight)
        else:
            assert False

    def get_increase_weight(self):
        initial_val = 1.0
        final_val = 1e-3
        current_iter = self.current_iter - self.increase_start_iter

        if self.increase_type == 'exp':
            ratio = 1.0 - math.exp(
                (current_iter / self.increase_iter) *
                math.log(final_val / (initial_val + 1e-6))) * initial_val
        else:
            ratio = 1.0 * current_iter / self.increase_iter
        return self.initial_weight + (self.final_weight -
                                      self.initial_weight) * ratio

    def step(self, current_iter=None):
        if not self.update_weight or self.fix_already:
            return

        if current_iter is not None:
            self.current_iter = current_iter

        if self.current_iter >= self.fix_start_iter:
            self.fix_already = True
            if hasattr(self.model, 'update_distill_weight'):
                self.model.update_distill_weight(self.final_weight)
        elif self.current_iter >= self.increase_start_iter:
            if hasattr(self.model, 'update_distill_weight'):
                self.model.update_distill_weight(self.get_increase_weight())

        self.current_iter += 1

    def get_weight(self):
        try:
            weight = self.model.distill_weight
        except Exception:
            weight = 0.0
        return weight
