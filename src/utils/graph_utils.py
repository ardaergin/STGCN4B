from typing import Tuple, Union, Literal
import torch
from torch_geometric.utils import add_self_loops, degree, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# Return types
GSODense  = torch.Tensor
GSOCoo    = Tuple[torch.LongTensor, torch.Tensor]
GSOSparse = torch.Tensor  # torch.sparse_coo_tensor

def calc_gso_edge(
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        gso_type: str,
        device: torch.device,
        return_format: Literal["dense", "coo", "sparse"] = "dense",
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