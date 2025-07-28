from typing import Mapping, Any, Dict, List
import torch
from torch import nn, Tensor
from typing import Mapping
from ..homogeneous.layers import TemporalConvLayer, Align
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv


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
        self.Kt = Kt
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
        T_in = next(iter(x_dict.values())).shape[2]
        T_out = T_in - (self.Kt - 1)

        if T_out <= 0:
            raise ValueError(
                f"History size {T_in} is too small for temporal kernel size {self.Kt}."
            )

        for ntype, x in x_dict.items():
            if ntype in self.static_ntypes:
                # Align expects (B, C, T, N), perfect fit
                aligned_x = self.blocks[ntype](x)
                # Align layer changes channels but not the time dimension.
                # Manually slice the time dimension to match the convolved tensors.
                out_dict[ntype] = aligned_x[:, :, :T_out, :]
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
            in_channels=ntype_channels_mid['room'],
            out_channels=ntype_channels_mid['room'],
            bias=bias,
            add_self_loops=True
        )
        convs[('room', 'adjacent_vertical', 'room')] = GCNConv(
            in_channels=ntype_channels_mid['room'],
            out_channels=ntype_channels_mid['room'],
            bias=bias,
            add_self_loops=True
        )

        # 2. Hierarchical/Structural relationships
        convs[('device', 'contained_in', 'room')] = SAGEConv(
            in_channels=(ntype_channels_mid['device'], ntype_channels_mid['room']),
            out_channels=ntype_channels_mid['room'],
            bias=bias
        )
        convs[('property', 'measured_by', 'device')] = SAGEConv(
            in_channels=(ntype_channels_mid['property'], ntype_channels_mid['device']),
            out_channels=ntype_channels_mid['device'],
            bias=bias
        )

        # 3. Weather influence relationships
        convs[('outside', 'influences', 'room')] = SAGEConv(
            in_channels=(ntype_channels_mid['outside'], ntype_channels_mid['room']),
            out_channels=ntype_channels_mid['room'],
            bias=bias
        )


        # 4. Time influence relationships
        convs[('time', 'affects', 'room')] = SAGEConv(
            in_channels=(ntype_channels_mid['time'], ntype_channels_mid['room']),
            out_channels=ntype_channels_mid['room'],
            bias=bias
        )
        convs[('time', 'affects', 'device')] = SAGEConv(
            in_channels=(ntype_channels_mid['time'], ntype_channels_mid['device']),
            out_channels=ntype_channels_mid['device'],
            bias=bias
        )
        convs[('time', 'affects', 'property')] = SAGEConv(
            in_channels=(ntype_channels_mid['time'], ntype_channels_mid['property']),
            out_channels=ntype_channels_mid['property'],
            bias=bias
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
            edge_index_dict: Dict[tuple, Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        # 1. First temporal layer
        x_dict_after_temp1 = self.temp1(x_dict)
        T_out_temp1 = next(iter(x_dict_after_temp1.values())).shape[2]
        
        # 2. Prepare separate dictionaries for edge_index and edge_weight
        edge_index_for_conv = {}
        edge_weight_for_conv = {}
        for etype, ew in edge_index_dict.items():
            edge_index_for_conv[etype] = ew['index'].long()
            if 'weight' in ew and ew['weight'] is not None:
                edge_weight_for_conv[etype] = ew['weight'].squeeze()

        # 3. Apply spatial layer at each time step
        out_slices = []
        for t in range(T_out_temp1):
            flat_x_t = {
                ntype: x[:, :, t, :].permute(0, 2, 1).reshape(-1, x.shape[1])
                for ntype, x in x_dict_after_temp1.items()
            }

            # This is the canonical way to call HeteroConv with weights
            flat_out_t = self.hetero_conv(
                x_dict=flat_x_t,
                edge_index_dict=edge_index_for_conv,
                edge_weight=edge_weight_for_conv
            )

            # Merge the GNN output back into the full feature set
            for ntype, out_feat in flat_out_t.items():
                flat_x_t[ntype] = out_feat

            # Un-flatten the complete feature set for this time step
            slice_out_t = {
                ntype: x_flat.view(x_dict_after_temp1[ntype].shape[0],      # B
                                  x_dict_after_temp1[ntype].shape[3],      # N
                                  -1)                                      # C
                           .permute(0, 2, 1)                               # -> (B, C, N)
                for ntype, x_flat in flat_x_t.items()
            }
            out_slices.append(slice_out_t)

        # 4. Stack time slices
        x_dict_after_spatial = {
            ntype: torch.stack([s[ntype] for s in out_slices], dim=2)
            for ntype in x_dict_after_temp1.keys()
        }

        # 5. Activation
        x_dict_after_relu = {k: self.relu(v) for k, v in x_dict_after_spatial.items()}
        
        # 6. Second temporal layer
        x_dict_after_temp2 = self.temp2(x_dict_after_relu)
        
        # 7. LayerNorm + Dropout
        final_x_dict = {}
        for ntype, x in x_dict_after_temp2.items():
            x_perm = x.permute(0, 2, 3, 1)
            x_norm = self.norms[ntype](x_perm)
            final_x_dict[ntype] = self.dropout(x_norm).permute(0, 3, 1, 2)
            
        return final_x_dict