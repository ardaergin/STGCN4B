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


class WeightedHeteroConv(nn.Module):
    """Custom heterogeneous convolution that properly handles edge weights."""
    
    def __init__(self, convs, aggr='sum'):
        super().__init__()
        # Convert edge type tuples to strings for ModuleDict
        self.convs = nn.ModuleDict()
        self.edge_type_map = {}  # Maps string keys back to edge type tuples
        
        for edge_type, conv in convs.items():
            # Create a string key from the edge type tuple
            key = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            self.convs[key] = conv
            self.edge_type_map[key] = edge_type
            
        self.aggr = aggr
    
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[tuple, Any],
    ) -> Dict[str, Tensor]:
        
        out_dict = {}
        
        for key, conv in self.convs.items():
            edge_type = self.edge_type_map[key]
            src, _, dst = edge_type
            
            edge_data = edge_index_dict.get(edge_type)
            if edge_data is None:
                continue
                
            # Handle edge data - could be just edge_index or (edge_index, edge_weight)
            if isinstance(edge_data, tuple) and len(edge_data) == 2:
                edge_index, edge_weight = edge_data
            else:
                edge_index = edge_data
                edge_weight = None
            
            # Get node features
            x_src = x_dict.get(src)
            x_dst = x_dict.get(dst)
            
            # For GCNConv, we need to pass edge_weight explicitly
            if isinstance(conv, GCNConv):
                # GCNConv with self-loops expects x to be the dst features
                out = conv(x_dst, edge_index, edge_weight=edge_weight)
            else:
                # SAGEConv and others can handle the tuple format
                if edge_weight is not None:
                    kwargs = {'edge_index': edge_index, 'edge_attr': edge_weight}
                else:
                    kwargs = {'edge_index': edge_index}
                
                if x_src is None:
                    out = conv(x_dst, **kwargs)
                else:
                    out = conv((x_src, x_dst), **kwargs)
            
            # Aggregate outputs for destination nodes
            if dst in out_dict:
                # Simple aggregation
                if self.aggr == 'sum':
                    out_dict[dst] = out_dict[dst] + out
                elif self.aggr == 'mean':
                    # Keep track of count for mean
                    if not hasattr(self, '_counts'):
                        self._counts = {}
                    if dst not in self._counts:
                        self._counts[dst] = 1
                        out_dict[dst] = out
                    else:
                        self._counts[dst] += 1
                        out_dict[dst] = out_dict[dst] + out
                elif self.aggr == 'max':
                    out_dict[dst] = torch.max(out_dict[dst], out)
                else:
                    raise ValueError(f"Unknown aggregation: {self.aggr}")
            else:
                out_dict[dst] = out
        
        # Apply mean aggregation if needed
        if self.aggr == 'mean' and hasattr(self, '_counts'):
            for dst, count in self._counts.items():
                out_dict[dst] = out_dict[dst] / count
            self._counts.clear()
        
        # Add nodes that were not updated
        for node_type, x in x_dict.items():
            if node_type not in out_dict:
                out_dict[node_type] = x
                
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

        # Use our custom heterogeneous convolution
        self.hetero_conv = WeightedHeteroConv(convs, aggr=aggr)
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
        # Temporal layer 1
        x_dict_after_temp1 = self.temp1(x_dict) # (B, C_mid, T_mid, N)
        
        # We get the shape parameters from the output of the first temporal block
        T_out_temp1 = next(iter(x_dict_after_temp1.values())).shape[2]
        
        # Prepare the edge data for HeteroConv *once* before the loop.
        # The graph structure is static across time steps.
        batched_edges_for_conv = {}
        for etype, ew in edge_index_dict.items():
            idx = ew['index'].long()
            wt = ew.get('weight')

            if wt is None:
                batched_edges_for_conv[etype] = idx
            else:
                # Ensure the weight tensor is 1D
                batched_edges_for_conv[etype] = (idx, wt.squeeze())

        # We will accumulate the processed slices and stack on the time dim later
        out_slices: List[Dict[str, Tensor]] = []
        
        for t in range(T_out_temp1):
            # a. Get the full feature set for time t
            flat_x_t = {
                ntype: x[:, :, t, :].permute(0, 2, 1).reshape(-1, x.shape[1])
                for ntype, x in x_dict_after_temp1.items()
            }

            # b. Perform spatial convolution using our custom HeteroConv
            flat_out_t = self.hetero_conv(flat_x_t, batched_edges_for_conv)

            # c. Un-flatten for this time step
            slice_out_t = {
                ntype: x_flat.view(x_dict_after_temp1[ntype].shape[0],      # B
                                x_dict_after_temp1[ntype].shape[3],      # N
                                -1)                                      # C
                        .permute(0, 2, 1)                               # -> (B, C, N)
                for ntype, x_flat in flat_out_t.items()
            }
            out_slices.append(slice_out_t)

        # 4. Stack the complete time slices to reconstruct the temporal dimension
        x_dict_after_spatial = {
            ntype: torch.stack([s[ntype] for s in out_slices], dim=2)
            for ntype in x_dict_after_temp1.keys()
        }

        # 5. ReLU activation
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