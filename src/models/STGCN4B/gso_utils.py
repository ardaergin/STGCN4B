from typing import Any, Dict, Literal, Tuple, List, Union
import torch
from torch_geometric.utils import add_self_loops, degree, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dense_to_sparse

import logging; logger = logging.getLogger(__name__)


# Return types
GSODense    = torch.Tensor
GSOCoo      = Tuple[torch.LongTensor, torch.Tensor]
GSOSparse   = torch.Tensor  # torch.sparse_coo_tensor
GSOType     = Union[GSODense, GSOCoo, GSOSparse]
GSOList     = List[GSOType]


def create_gso(
        args: Any, 
        device: torch.device,
        n_nodes: int,
        adj_matrix: torch.Tensor,
        masked_adj_matrices: Dict[str, torch.Tensor] = None,
        return_format: Literal["dense", "coo", "sparse"] = "dense",
        transpose: bool = False
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
        return_format       = return_format,
        transpose           = transpose
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
                return_format   = return_format,
                transpose       = transpose
            )
            masked_gsos.append(gso)
        # Pad with the static GSO if not enough dynamic ones are available
        while len(masked_gsos) < args.stblock_num:
            masked_gsos.append(static_gso)
        return masked_gsos
    
    else:
        raise ValueError(f"Unknown gso_mode: {args.gso_mode!r}.")


def calc_gso_edge(
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        gso_type: str,
        device: torch.device,
        return_format: Literal["dense", "coo", "sparse"] = "dense",
        transpose: bool = False,
) -> Union[GSODense, GSOCoo, GSOSparse]:
    """
    Compute a Graph Shift Operator (GSO).

    Provides various normalization options, with gso_type ∈ {
      'sym_norm_adj', 'sym_renorm_adj', 'sym_norm_lap', 'sym_renorm_lap',
      'rw_norm_adj',  'rw_renorm_adj',  'rw_norm_lap',  'rw_renorm_lap',
      'col_renorm_adj',
      'no_norm_only_self_loop'
    }

    return_format:
      - "dense"  -> returns a dense (num_nodes x num_nodes) tensor [default]
      - "coo"    -> returns (edge_index: LongTensor [2, E], edge_weight: Tensor [E])
      - "sparse" -> returns a torch.sparse_coo_tensor (num_nodes x num_nodes)
    """
        
    if gso_type == 'no_norm_only_self_loop':
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight,
            fill_value=1.0,
            num_nodes=num_nodes
        )

    else:
        # 1) Handle adjacency vs Laplacian
        is_lap = gso_type.endswith('_lap')
        renorm = 'renorm' in gso_type
        if gso_type.startswith('sym'):
            base = 'sym'
        elif gso_type.startswith('rw'):
            base = 'rw'
        elif gso_type.startswith('col'):
            base = 'col'

        if not is_lap:
            # --- adjacency normalization ---
            if base == 'sym':
                # symmetric renormalized adj: D^-½ (A + I) D^-½
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight,
                    num_nodes=num_nodes,
                    add_self_loops=renorm,
                    improved=False  # use 1 in I; set True for “2I” trick
                )
            else: # 'rw' and 'col'
                if renorm:
                    edge_index, edge_weight = add_self_loops(
                        edge_index, edge_weight,
                        fill_value=1.0, 
                        num_nodes=num_nodes
                    )
                if base == 'rw':
                    # Row normalization (out-degree)
                    # D⁻¹ (A + I) if renorm else D⁻¹ A
                    row, _ = edge_index
                    deg = degree(row, num_nodes=num_nodes, dtype=edge_weight.dtype)
                    deg_inv = deg.pow(-1)
                    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
                    edge_weight = deg_inv[row] * edge_weight
                
                elif base == 'col':
                    # Column normalization (in-degree)
                    _, col = edge_index
                    deg = degree(col, num_nodes=num_nodes, dtype=edge_weight.dtype)
                    deg_inv = deg.pow(-1)
                    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
                    edge_weight = edge_weight * deg_inv[col]
        
        else:
            # --- laplacian normalization ---
            norm_type = 'sym' if base == 'sym' else 'rw'
            if renorm:
                edge_index, edge_weight = add_self_loops(
                    edge_index,
                    edge_weight,
                    fill_value=1.0,
                    num_nodes=num_nodes
                )
            # now compute (I - Â) or D^-½(D - Â)D^-½
            norm_type = 'sym' if base == 'sym' else 'rw'
            edge_index, edge_weight = get_laplacian(
                edge_index,
                edge_weight,
                normalization=norm_type,
                num_nodes=num_nodes
            )

    # Apply transpose at the very end, after normalization
    if transpose:
        edge_index = edge_index[[1, 0]]
    
    # Return in the requested format
    if return_format == "coo":
        return edge_index.to(device), edge_weight.to(device)

    sparse = torch.sparse_coo_tensor(
        edge_index, edge_weight,
        (num_nodes, num_nodes),
        device=device
    )

    if return_format == "sparse":
        return sparse

    # default: dense
    return sparse.to_dense()