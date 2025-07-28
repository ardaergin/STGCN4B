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
        convs[('room', 'adjacent_horizontal', 'room')] = GCNConv(
            ntype_channels_mid['room'], ntype_channels_mid['room'], bias=bias
        )
        convs[('room', 'adjacent_vertical', 'room')] = GCNConv(
            ntype_channels_mid['room'], ntype_channels_mid['room'], bias=bias
        )

        # 2. Hierarchical/Structural relationships
        convs[('device', 'contained_in', 'room')] = GCNConv(
            ntype_channels_mid['device'], ntype_channels_mid['room'], bias=bias, add_self_loops=False
        )
        convs[('property', 'measured_by', 'device')] = GCNConv(
            ntype_channels_mid['property'], ntype_channels_mid['device'], bias=bias, add_self_loops=False
        )

        # 3. Weather influence relationships
        convs[('outside', 'influences', 'room')] = GCNConv(
            ntype_channels_mid['outside'], ntype_channels_mid['room'], bias=bias, add_self_loops=False
        )

        # 4. Time influence relationships
        convs[('time', 'affects', 'room')] = GCNConv(
            ntype_channels_mid['time'], ntype_channels_mid['room'], bias=bias, add_self_loops=False
        )
        convs[('time', 'affects', 'device')] = GCNConv(
            ntype_channels_mid['time'], ntype_channels_mid['device'], bias=bias, add_self_loops=False
        )
        convs[('time', 'affects', 'property')] = GCNConv(
            ntype_channels_mid['time'], ntype_channels_mid['property'], bias=bias, add_self_loops=False
        )

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
            edge_index_dict: Dict[tuple, Dict[str, Tensor]] # Renamed for clarity
    ) -> Dict[str, Tensor]:
        # Temporal layer 1
        x_dict = self.temp1(x_dict)  # (B, C_mid, T_mid, N)
        
        # We get the shape parameters from the output of the first temporal block
        T_mid = next(iter(x_dict.values())).shape[2]

        # Prepare the edge data for HeteroConv *once* before the loop.
        # The graph structure is static across time steps.
        
        batched_edges_for_conv: Dict[tuple, torch.Tensor] = {
            etype: ew['index']          # <<< WEIGHTS DROPPED HERE
            for etype, ew in edge_index_dict.items()
        }

        # for etype, ew in edge_index_dict.items():
        #     idx = ew['index']
        #     wt = ew.get('weight') # Use .get() for safety if weight is sometimes missing
        #     if wt is None:
        #         batched_edges_for_conv[etype] = idx
        #     else:
        #         batched_edges_for_conv[etype] = (idx, wt)

        # We will accumulate the processed slices and stack on the time dim later
        out_slices: List[Dict[str, Tensor]] = []
        
        for t in range(T_mid):
            # 1. Build flattened node feature dict for the current time step `t`
            flat_x_t: Dict[str, Tensor] = {
                ntype: x[:, :, t, :].permute(0, 2, 1).reshape(-1, x.shape[1]).contiguous()
                for ntype, x in x_dict.items()
            }
            
            # 2. Perform spatial message passing on the time slice `t`
            #    using the pre-batched edge information.
            flat_out_t = self.hetero_conv(flat_x_t, batched_edges_for_conv)
            
            # 3. Un-flatten the output back to (B, C, N) and append for later stacking
            slice_out = {}
            for ntype, out in flat_out_t.items():
                B, N = x_dict[ntype].shape[0], x_dict[ntype].shape[3]
                C_out = out.shape[1]
                slice_out[ntype] = out.view(B, N, C_out).permute(0, 2, 1) # (B, C_out, N)
            out_slices.append(slice_out)
        
        # 4. Stack time slices to reconstruct the temporal dimension
        x_dict = {
            ntype: torch.stack([s[ntype] for s in out_slices], dim=2)
            for ntype in x_dict.keys()
        }
        
        x_dict = {k: self.relu(v) for k, v in x_dict.items()}
        
        # Temporal layer 2
        x_dict = self.temp2(x_dict)

        # 5. LayerNorm + Dropout
        for ntype, x in x_dict.items():
            x_perm = x.permute(0, 2, 3, 1)
            x_norm = self.norms[ntype](x_perm)
            x_dict[ntype] = self.dropout(x_norm).permute(0, 3, 1, 2)
            
        return x_dict