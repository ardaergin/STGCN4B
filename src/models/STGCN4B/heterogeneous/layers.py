from typing import Mapping, Any, Dict, List
import torch
from torch import nn, Tensor
from typing import Mapping
from ..homogeneous.layers import TemporalConvLayer, Align
from torch_geometric.nn import HeteroConv, GCNConv


class HeteroTemporalBlock(nn.Module):
    """Applies TemporalConvLayer to temporal nodes and a simple projection to static nodes."""
    def __init__(
            self,
            ntype_channels_in:      Mapping[str, int],
            ntype_channels_out:     Mapping[str, int],
            Kt:                     int,
            act_func:               str = "glu",
            static_ntypes:          List[str] = ('room', 'device')
    ):
        super().__init__()
        self.blocks = nn.ModuleDict()
        self.static_ntypes = static_ntypes
        for ntype, Cin in ntype_channels_in.items():
            Cout = ntype_channels_out[ntype]
            if ntype in self.static_ntypes:
                self.blocks[ntype] = Align(Cin, Cout)
            else:
                self.blocks[ntype] = TemporalConvLayer(
                    Kt                  = Kt,
                    c_in                = Cin,
                    c_out               = Cout,
                    n_vertex            = 1, # n_vertex is not used in the calculation, can be 1
                    act_func            = act_func
                )

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            x_dict (Dict[str, Tensor]): Dict of tensors with shape (B, C, T, N)
        """
        out_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.static_ntypes:
                # Align expects (B, C, T, N), perfect fit
                out_dict[ntype] = self.blocks[ntype](x)
            else:
                # Reshape for TemporalConvLayer: (B,C,T,N) -> (B*N,C,T,1)
                B, C, T, N = x.shape
                x_reshaped = x.permute(0, 3, 1, 2).reshape(B * N, C, T, 1)
                output_reshaped = self.blocks[ntype](x_reshaped) # (B*N, Cout, Tout, 1)
                _, Cout, Tout, _ = output_reshaped.shape
                # Reshape back: (B*N,Cout,Tout,1) -> (B,N,Cout,Tout) -> (B,Cout,Tout,N)
                out_dict[ntype] = output_reshaped.view(B, N, Cout, Tout).permute(0, 2, 3, 1)
        return out_dict


class HeteroSTBlock(nn.Module):
    """T -> S -> T -> (LayerNorm, Dropout) for heterogeneous graphs."""
    def __init__(
            self,
            Kt:                     int,
            ntype_channels_in:      Mapping[str, int],
            ntype_channels_mid:     Mapping[str, int],
            ntype_channels_out:     Mapping[str, int],
            metadata:               tuple,
            act_func:               str = "glu",
            bias:                   bool = True,
            droprate:               float = 0.0,
            aggr:                   str = "sum"
    ):
        super().__init__()
        
        # Temporal layer 1
        self.temp1 = HeteroTemporalBlock(ntype_channels_in, ntype_channels_mid, Kt, act_func)

        ##### Spatial layer #####
        # Build one GNN layer per edge-type that should perform message passing
        convs = {}
        
        # 1. Spatial relationships between rooms (simple diffusion)
        convs[('room', 'adjacent_horizontal', 'room')] = GCNConv(-1, ntype_channels_mid['room'], bias=bias)
        convs[('room', 'adjacent_vertical', 'room')] = GCNConv(-1, ntype_channels_mid['room'], bias=bias)
        
        # 2. Hierarchical/Structural relationships
        convs[('device', 'contained_in', 'room')] = GCNConv(-1, ntype_channels_mid['room'], bias=bias, add_self_loops=False)
        convs[('property', 'measured_by', 'device')] = GCNConv(-1, ntype_channels_mid['device'], bias=bias, add_self_loops=False)
        
        # 3. Weather influence relationships
        convs[('outside', 'influences', 'room')] = GCNConv(-1, ntype_channels_mid['room'], bias=bias, add_self_loops=False)
        
        # 4. Time influence relationships
        convs[('time', 'affects', 'room')] = GCNConv(-1, ntype_channels_mid['room'], bias=bias, add_self_loops=False)
        convs[('time', 'affects', 'device')] = GCNConv(-1, ntype_channels_mid['device'], bias=bias, add_self_loops=False)
        convs[('time', 'affects', 'property')] = GCNConv(-1, ntype_channels_mid['property'], bias=bias, add_self_loops=False)

        self.hetero_conv = HeteroConv(convs, aggr=aggr)
        ##### End of spatial layer #####
        
        # Activation function
        self.relu = nn.ReLU()

        # Temporal layer 2
        self.temp2 = HeteroTemporalBlock(ntype_channels_mid, ntype_channels_out, Kt, act_func)
        
        # LayerNorm and Dropout
        # NOTE: LayerNorm expects (..., C) so we will permute before applying it
        self.norms = nn.ModuleDict({
            ntype: nn.LayerNorm(C_out) for ntype, C_out in ntype_channels_out.items()
        })
        self.dropout = nn.Dropout(droprate)

    def forward(
            self, 
            x_dict: Dict[str, Tensor], 
            edge_dict: Dict[tuple, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        # Temporal layer 1
        x_dict = self.temp1(x_dict)  # (B, C_mid, T_mid, N)
        
        B, T_mid = next(iter(x_dict.values())).shape[:2]
        node_count = {nt: x.shape[3] for nt, x in x_dict.items()}   # N per type
        
        # We will accumulate the processed slices and stack on time dim later
        out_slices: List[Dict[str, Tensor]] = []
        
        for t in range(T_mid):
            # 1a. Build flattened node feature dict
            flat_x: Dict[str, Tensor] = {}
            for ntype, x in x_dict.items():       # x: (B, C, T, N)
                B_, C_, N_ = x.shape[0], x.shape[1], x.shape[3]
                flat_x[ntype] = (
                    x[:, :, t, :]                 # (B, C, N)
                    .permute(0, 2, 1)             # (B, N, C)
                    .reshape(B_ * N_, C_)         # (B·N, C)
                    .contiguous()
                )

            # 1b.  Repeat edges for B graphs
            batched_edges: Dict[tuple, Any] = {}
            for etype, ew in edge_dict.items():
                base_idx  = ew["index"]           # (2, E)
                base_wt   = ew["weight"]          # (E) or None
                src, _, dst = etype
                N_src, N_dst = node_count[src], node_count[dst]

                # replicate indices
                rep_idx = base_idx.unsqueeze(0).repeat(B, 1, 1)
                offset  = torch.arange(B, device=base_idx.device).view(B, 1)
                rep_idx[:, 0, :] += offset * N_src
                rep_idx[:, 1, :] += offset * N_dst
                rep_idx = rep_idx.view(2, -1)

                if base_wt is None:                           # un‑weighted edge
                    batched_edges[etype] = rep_idx
                else:                                         # weighted edge
                    rep_wt = base_wt.repeat(B, 1).view(-1)
                    batched_edges[etype] = (rep_idx, rep_wt)
            
            # 1c.  Spatial message passing
            flat_out = self.hetero_conv(flat_x, batched_edges)  # each (B·N, C_out)
            
            # 1d.  Un‑flatten back to (B, C_out, N)
            slice_out = {}
            for ntype, out in flat_out.items():
                N_ = node_count[ntype]
                C_out = out.shape[1]
                slice_out[ntype] = (
                    out.view(B, N_, C_out)
                    .permute(0, 2, 1)  # (B, C_out, N)
                )
            
            out_slices.append(slice_out)
        
        # 2.  Stack along time dim & T step 2
        x_dict = {ntype: torch.stack([s[ntype] for s in out_slices], dim=2)
                for ntype in x_dict}
        
        x_dict = {k: self.relu(v) for k, v in x_dict.items()}
        
        # Temporal layer 2
        x_dict = self.temp2(x_dict)

        # 3.  LayerNorm + Dropout
        for ntype, x in x_dict.items():           # (B, C_out, T_out, N)
            x_perm = x.permute(0, 2, 3, 1)        # (B, T, N, C)
            x_norm = self.norms[ntype](x_perm)
            x_dict[ntype] = self.dropout(x_norm).permute(0, 3, 1, 2)

        return x_dict