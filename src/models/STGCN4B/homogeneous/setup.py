from typing import Any, Dict, Literal, Tuple, List, Union
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from ....utils.graph_utils import calc_gso_edge

import logging; logger = logging.getLogger(__name__)


# Return types
GSODense  = torch.Tensor
GSOCoo    = Tuple[torch.LongTensor, torch.Tensor]
GSOSparse = torch.Tensor  # torch.sparse_coo_tensor

GSOType = Union[GSODense, GSOCoo, GSOSparse]
GSOList = List[GSOType]

def create_gso(
        args: Any, 
        device: torch.device,
        n_nodes: int,
        adj_matrix: torch.Tensor,
        masked_adj_matrices: Dict[str, torch.Tensor] = None,
        return_format: Literal["dense", "coo", "sparse"] = "dense",
) -> Union[GSOType, GSOList]:
    """
    Creates the Graph Shift Operator (GSO) for the model.
    
    Args:
        args (Any): The argument parser containing model configuration. Must include:
            - gso_type: Type of GSO to create (e.g., 'rw_norm_adj', 'rw_renorm_adj').
            - gso_mode: Mode of GSO ('static' or 'dynamic').
            - adjacency_type: Type of adjacency matrix (e.g., 'weighted').
            - stblock_num: Number of STGCN blocks (used for dynamic GSOs).
        device (torch.device): The device to place the GSO on.
        n_nodes (int): The number of nodes in the graph.
        adj_matrix (torch.Tensor): The main adjacency matrix for the graph.
        masked_adj_matrices (Dict[str, torch.Tensor], optional): Additional masked adjacency matrices for dynamic GSOs.    
        return_format: 
            - "dense"  -> returns a dense (N x N) tensor [default]
            - "coo"    -> returns (edge_index [2,E], edge_weight [E])
            - "sparse" -> returns torch.sparse_coo_tensor (N x N)
    
    Returns:
        If gso_mode == "static": a single GSO in `return_format`.
        If gso_mode == "dynamic": a list of GSOs in `return_format` with length == stblock_num.
    """
    # Adjacency type check
    if args.adjacency_type == "weighted" and args.gso_type not in ("rw_norm_adj", "rw_renorm_adj"):
        logger.warning("For weighted adjacency, the recommended gso_type is 'rw_norm_adj' or 'rw_renorm_adj'.")
    
    # Create the base static GSO from the main adjacency matrix
    static_A = adj_matrix
    edge_index, edge_weight = dense_to_sparse(static_A)
    logger.info(f"edge_index shape: {edge_index.shape}")
    logger.info(f"edge_weight shape: {edge_weight.shape}")
    
    static_gso = calc_gso_edge(
        edge_index, edge_weight, 
        num_nodes           = n_nodes,
        gso_type            = args.gso_type,
        device              = device,
        return_format       = return_format
    )
    if args.gso_mode == "static":
        return static_gso
    
    # Build masked GSOs for information propagation
    elif args.gso_mode == "dynamic":
        masked_gsos = []
        masked_matrices = list(masked_adj_matrices.values())
        for mat in masked_matrices[:args.stblock_num]:
            edge_index, edge_weight = dense_to_sparse(mat)
            gso = calc_gso_edge(
                edge_index, edge_weight, 
                num_nodes       = n_nodes,
                gso_type        = args.gso_type, 
                device          = device,
                return_format   = return_format
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