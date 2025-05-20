import torch
from torch_geometric.utils import add_self_loops, degree, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def calc_gso_edge(edge_index: torch.LongTensor,
                  edge_weight: torch.Tensor,
                  num_nodes: int,
                  gso_type: str,
                  device: torch.device):
    """
    Replace calc_gso + calc_chebynet_gso + cnv_sparse_mat_to_coo_tensor.

    gso_type ∈ {
      'sym_norm_adj', 'sym_renorm_adj', 'sym_norm_lap', 'sym_renorm_lap',
      'rw_norm_adj',  'rw_renorm_adj',  'rw_norm_lap',  'rw_renorm_lap'
    }
    """
    # 1) Handle adjacency vs Laplacian
    is_lap = gso_type.endswith('_lap')
    base = 'sym' if gso_type.startswith('sym') else 'rw'
    renorm = 'renorm' in gso_type

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
        else:
            # random-walk adj: D⁻¹ (A + I) if renorm else D⁻¹ A
            if renorm:
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight,
                    fill_value=1.0, num_nodes=num_nodes)
            row, _ = edge_index
            deg = degree(row, num_nodes=num_nodes)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = deg_inv[row] * edge_weight

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

    # Now return a dense GSO (so that your Cheb / GraphConv layers see a Tensor)
    return torch.sparse_coo_tensor(
        edge_index, edge_weight,
        (num_nodes, num_nodes),
        device=device
    ).to_dense()
