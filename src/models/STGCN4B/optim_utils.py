from typing import Any, Dict, Literal, Tuple, List, Union
import torch
import torch.nn as nn

import logging; logger = logging.getLogger(__name__)


_NORM_TYPES = (
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
)
_GATE_LEAF_NAMES = {
    "g_p2d_dev", "g_d2p_prop", "g_d2r_room", "g_r2d_dev",
    "g_time2room", "g_outside2room",
}
def _collect_no_decay_param_names(model: nn.Module) -> set[str]:
    no_decay = set()
    
    if hasattr(model, "device_embedding"):
        no_decay.add("device_embedding.weight")
    
    # 1) all params directly inside normalization modules
    for mod_name, m in model.named_modules():
        if isinstance(m, _NORM_TYPES):
            for p_name, _ in m.named_parameters(recurse=False):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                no_decay.add(full)

    # 2) gate scalars by exact leaf name
    for n, _ in model.named_parameters():
        leaf = n.rsplit(".", 1)[-1]
        if leaf in _GATE_LEAF_NAMES:
            no_decay.add(n)

    # 3) singleton projections (time/outside) — exclude weight & bias
    for n, _ in model.named_parameters():
        if ".time_proj_room." in n or ".outside_proj." in n:
            no_decay.add(n)

    # 4) biases
    for n, _ in model.named_parameters():
        if n.endswith(".bias"):
            no_decay.add(n)

    return no_decay

def create_optimizer(args: Any, model: nn.Module) -> torch.optim.Optimizer:
    no_decay_names = _collect_no_decay_param_names(model)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n in no_decay_names else decay).append(p)

    param_groups = [
        {"params": decay,    "weight_decay": args.weight_decay_rate},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    opt = (
        torch.optim.AdamW(param_groups, lr=args.lr) if args.optimizer == "adamw"
        else torch.optim.Adam(param_groups, lr=args.lr) if args.optimizer == "adam"
        else torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9)
    )

    num_decay   = sum(p.numel() for p in decay)
    num_nodcay  = sum(p.numel() for p in no_decay)
    print(f"[WD groups] decay={num_decay:,} params, no_decay={num_nodcay:,} params")
    return opt

def create_scheduler(args: Any, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """Creates the learning rate scheduler."""
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    logger.info(f"Using scheduler: {scheduler.__class__.__name__} "
            f"with step_size={getattr(scheduler, 'step_size', 'N/A')} "
            f"and gamma={getattr(scheduler, 'gamma', 'N/A')}")
    return scheduler