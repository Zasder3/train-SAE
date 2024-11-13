# Adapted from open_clip: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py
import numpy as np
import torch

from train_sae.configs.base import RunConfig


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return 0.9 * base_lr * (step + 1) / warmup_length + 0.1 * base_lr


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def linear_decay_lr(optimizer, base_lr, warmup_length, steps):
    """
    Warmup the learning rate for `warmup_length` steps, hold it constant, then decay it
    linearly for the last 20% of training steps.

    Args:
        optimizer: The optimizer to adjust the learning rate of.
        base_lr: The initial learning rate.
        warmup_length: The number of warmup steps.
        steps: The total number of training steps.

    Returns:
        A function that adjusts the learning rate based on the current step.
    """

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        elif step < 0.8 * steps:
            lr = base_lr
        else:
            lr = base_lr * (1 - (step - 0.8 * steps) / (0.2 * steps))

        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def configure_scheduler(optimizer: torch.optim.Optimizer, config: RunConfig):
    """
    Configure the learning rate scheduler based on the configuration.

    Args:
        optimizer: The optimizer to adjust the learning rate of.
        config: The configuration object.

    Returns:
        A function that adjusts the learning rate based on the current step.
    """
    if config.lr_scheduler == "constant":
        return lambda step: None
    elif config.lr_scheduler == "cosine":
        return cosine_lr(optimizer, config.lr, config.lr_warmup_steps, config.num_steps)
    elif config.lr_scheduler == "linear_decay":
        return linear_decay_lr(
            optimizer, config.lr, config.lr_warmup_steps, config.num_steps
        )
    else:
        raise ValueError(f"Invalid learning rate schedule: {config.lr_schedule}")
