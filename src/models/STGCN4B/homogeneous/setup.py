from typing import Any, Dict
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from ....utils.graph_utils import calc_gso_edge

import logging; logger = logging.getLogger(__name__)


def create_gso(args: Any, input_dict: Dict[str, Any]) -> Any:
    """Creates the Graph Shift Operator (GSO) for the model."""    
    # Adjacency type check
    if args.adjacency_type == "weighted" and args.gso_type not in ("rw_norm_adj", "rw_renorm_adj"):
        raise ValueError("For weighted adjacency, gso_type must be 'rw_norm_adj' or 'rw_renorm_adj'.")

    # Create the base static GSO from the main adjacency matrix
    static_A = input_dict["adjacency_matrix"]
    edge_index, edge_weight = dense_to_sparse(static_A)
    logger.info(f"edge_index shape: {edge_index.shape}")
    logger.info(f"edge_weight shape: {edge_weight.shape}")
    
    static_gso = calc_gso_edge(
        edge_index, edge_weight, 
        num_nodes           = input_dict['n_nodes'],
        gso_type            = args.gso_type,
        device              = input_dict['device'],
    )
    if args.gso_mode == "static":
        return static_gso
    
    # Build masked GSOs for information propagation
    elif args.gso_mode == "dynamic":
        masked_gsos = []
        masked_matrices = list(input_dict.get("masked_adjacency_matrices", {}).values())
        for adj_matrix in masked_matrices[:args.stblock_num]:
            edge_index, edge_weight = dense_to_sparse(adj_matrix)
            gso = calc_gso_edge(
                edge_index, edge_weight, 
                num_nodes   = input_dict['n_nodes'],
                gso_type    = args.gso_type, 
                device      = input_dict['device']
            )
            masked_gsos.append(gso)
        # Pad with the static GSO if not enough dynamic ones are available
        while len(masked_gsos) < args.stblock_num:
            masked_gsos.append(static_gso)
        return masked_gsos
    else:
        raise ValueError(f"Unknown gso_mode: {args.gso_mode!r}.")

def create_optimizer(args: Any, model: nn.Module) -> torch.optim.Optimizer:
    """Creates the optimizer for the model."""
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_rate)
    
    logger.info(f"Using optimizer: {optimizer.__class__.__name__} "
                f"with lr={args.lr:.6f}, "
                f"weight_decay={args.weight_decay_rate:.6f}")
    return optimizer

def create_scheduler(args: Any, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    """Creates the learning rate scheduler."""
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    logger.info(f"Using scheduler: {scheduler.__class__.__name__} "
            f"with step_size={getattr(scheduler, 'step_size', 'N/A')} "
            f"and gamma={getattr(scheduler, 'gamma', 'N/A')}")
    return scheduler