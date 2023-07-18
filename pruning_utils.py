import torch
import math
import numpy as np


def refresh_pruning(model):
    for m in model.parameters():
        if hasattr(m, 'apply_mask'):
            m.apply_mask()


def apply_pruning(module, name, mask):
    param = getattr(module, name)

    def apply_mask():
        param.data *= mask.to(param.data.device, non_blocking=True)

    setattr(param, 'apply_mask', apply_mask)
    param.apply_mask()


def generate_mask(parameters: torch.Tensor, period: int):
    squeezed_parameters = parameters.squeeze()
    prime = math.gcd(squeezed_parameters.shape[-1], period) == 1
    shape = list(squeezed_parameters.shape)
    if not prime:
        shape[-1] += 1
    mask = torch.zeros(math.prod(shape), device=squeezed_parameters.device)
    mask[::period] = 1
    mask = mask.view(shape)
    if not prime:
        mask = mask.view(-1, mask.shape[-1])[:, :-1].view(squeezed_parameters.shape)
    mask = mask.view(parameters.shape)
    return mask


def generate_random_mask(parameters: torch.Tensor, rate: float):
    mask = torch.zeros(parameters.numel(), device=parameters.device)
    indices = np.random.choice(np.arange(len(mask)), int(len(mask) * rate), replace=False)
    mask[indices] = 1
    mask = mask.view(parameters.shape)
    return mask


def prune_network(model: torch.nn.Module, period: int, random_rate: float = 0., skip_first: bool = False):
    for m in model.modules():
        if skip_first and isinstance(m, torch.nn.Conv2d):
            if m.in_channels == 3:
                continue
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if random_rate != 0.:
                mask = generate_random_mask(m.weight, random_rate)
            else:
                mask = generate_mask(m.weight, period)
            apply_pruning(m, 'weight', mask)


def count_remaining_parameters(model: torch.nn.Module):
    remaining_count = 0
    total_count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):  # We don't count BatchNorm weights
            total_count += m.weight.numel()
            params = m.weight
            remaining_count += (params != 0).sum().item()
        elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Linear):
            total_count += m.bias.numel()
            remaining_count += m.bias.numel()
        else:
            pass
    return remaining_count, total_count


def remove_parameters(model: torch.nn.Module):
    for m in model.parameters():
        if hasattr(m, 'apply_mask'):
            m.apply_mask()
            delattr(m, 'apply_mask')
