# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimizers used when training pytorch models."""

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch.optim import Optimizer 
from torch.optim.lr_scheduler import LambdaLR


class ConstantLRSchedule(LambdaLR):
    """Constant learning rate schedule.
    
    Params:
    -------
    optimizer: torch.optim.Optimizer
        Wrapped optimizer.
    
    last_epoch: int, default=-1
        Index of last epoch.
    """
    def __init__(self, optimizer: Optimizer, last_epoch: int=-1):
        # NOTE: The multiplicative factor (lr_lambda) is set to 1.0 to keep a 
        # constant lr for all steps.
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, \
            last_epoch=last_epoch)


class WramupConstantSchedule(LambdaLR):
    """Constant learning rate schedule with initial linear warmup.

    Linearly increases learning rate from 0 to 1 over `warmup_steps` 
    steps, then stays at 1 afterwards.

    Params:
    -------
    optimizer: torch.optim.Optimizer
        Wrapped optimizer.

    warmup_steps: int
        Number of steps to use for warming up lr.
    
    last_epoch: int, default=-1
        Index of last epoch.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int=-1):
        self.warmup_steps = warmup_steps
        super(WramupConstantSchedule, self).__init__(optimizer, self.lr_lambda, \
            last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup then decay.
    
    Linearly increases learning rate from 0 to 1 over `warmup_steps` 
    steps. Then, linearly decays from 1 to 0 over the remaining 
    `total_steps - warmup_steps` steps. 

    Params:
    -------
    optimizer: torch.optim.Optimizer
        Wrapped optimizer.

    warmup_steps: int
        Number of steps to use for warming up lr.

    total_steps: int
        Total number of steps used for optimization.
    
    last_epoch: int, default=-1
        Index of last epoch.
    """
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int, 
                 total_steps: int, last_epoch: int=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, \
            last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return max(0.0, float(self.total_steps - step) / float(
            max(1.0, self.total_steps - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    
    Linearly increases learning rate from 0 to 1 over `warmup_steps` 
    steps. Then, decreases learning rate from 1 to 0 over remaining 
    `total_steps - warmup_steps` steps following a cosine curve. If 
    `cycles` (default=0.5) is different from default, learning rate 
    follows `cycle` times a cosine function after warmup.
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, 
                 total_steps: int, cycles: float=0.5, last_epoch: int=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, \
            last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """Linear warmup and then cosine cycles with hard restarts.
    
    Linearly increases learning rate from 0 to 1 over `warmup_steps` 
    steps. If `cycles` (default=1.0) is different from default, the lr 
    follows `cycles` times a cosine decaying learning rate (with hard 
    restarts).
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, 
                 total_steps: int, cycles: float=1., last_epoch: int=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, \
            self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps))
        if progress >= 1.0:
            return 0.0 # if lr progress > 1.0 after the initial warmup, reset lr
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))


class AdamW(Optimizer):
    """Implements Adam algorithm with weight decay fix.

    Params:
    -------
    params: iterable
        An iterable of :class:`torch.Tensor` s or :class:`dict`s. 
        Specifies what Tensors should be optimized.

    lr: float, default=1e-3
        Learning rate.
    
    betas: tuple of floats, default=(0.9, 0.999) 
        Adams beta parameters (b1, b2).
    
    eps: float, default=1e-6
        Adams epsilon.
    
    weight_decay: float, default=0.0 
        Weight decay.
    
    correct_bias: bool, default=True
        If False, avoids correcting bias in Adam (e.g. like in Bert TF 
        repository).
    """
    
    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 lr: float=1e-3,
                 betas: Tuple[float, float]=(0.9, 0.999),
                 eps: float=1e-6,
                 weight_decay: float=0.0,
                 correct_bias: bool=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]]=None) -> Optional[float]:
        """Performs a single optimization step (parameter update).

        Params:
        -------
        closure: callable, optional, default=None
            A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, '
                                       'please consider SparseAdam instead.')

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss