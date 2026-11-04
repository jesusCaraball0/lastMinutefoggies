"""Explicit AdamW parameter grouping for controlled regularization experiments."""
from torch import nn


def weight_decay_groups(model, weight_decay, exclude_norm_bias=False):
    """Optionally decay only matrices/kernels; preserve every trainable parameter."""
    decay, no_decay, decay_names, no_decay_names = [], [], [], []
    norm_ids = {id(p) for module in model.modules()
                if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm))
                for p in module.parameters(recurse=False)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = exclude_norm_bias and (id(parameter) in norm_ids or name.endswith('.bias'))
        (no_decay if excluded else decay).append(parameter)
        (no_decay_names if excluded else decay_names).append(name)
    groups = [{'params': decay, 'weight_decay': weight_decay}]
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return groups, {'decay': decay_names, 'no_decay': no_decay_names}
