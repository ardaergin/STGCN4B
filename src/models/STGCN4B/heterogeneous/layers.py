from typing import Mapping, Dict
import torch
from torch import nn, Tensor
from typing import Mapping
from ..homogeneous.layers import TemporalConvLayer
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv


class HeteroTemporalBlock(nn.Module):
    """
    Applies a TemporalConvLayer to all node types in the graph, processing
    each node's features for temporal patterns.
    """
    def __init__(
            self,
            ntype_channels_in:      Mapping[str, int],
            ntype_channels_out:     Mapping[str, int],
            Kt:                     int,
            act_func:               str = "glu",
    ):
        super().__init__()
        self.Kt = Kt
        self.blocks = nn.ModuleDict()
        
        # Create a TemporalConvLayer for every node type
        for ntype, Cin in ntype_channels_in.items():
            Cout = ntype_channels_out[ntype]
            self.blocks[ntype] = TemporalConvLayer(
                Kt=Kt,
                c_in=Cin,
                c_out=Cout,
                n_vertex=1,  # n_vertex is not used in the calculation, can be 1
                act_func=act_func
            )
    
    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            x_dict (Dict[str, Tensor]): Dict of tensors with shape (B, C, T, N)
        """
        out_dict = {}
        
        # Apply the temporal convolution to each node type's feature tensor
        for ntype, x in x_dict.items():
            # Reshape for TemporalConvLayer: (B, C, T, N) -> (B*N, C, T, 1)
            B, C, T, N = x.shape
            x_reshaped = x.permute(0, 3, 1, 2).reshape(B * N, C, T, 1)
            
            # Apply the temporal convolution
            output_reshaped = self.blocks[ntype](x_reshaped)  # (B*N, Cout, Tout, 1)
            
            # Reshape back: (B*N, Cout, Tout, 1) -> (B, N, Cout, Tout) -> (B, Cout, Tout, N)
            _, Cout, Tout, _ = output_reshaped.shape
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
            aggr:                   str = "sum",
            heads:                  int = 4,
            gconv_type_p2d:          str = "sage",
            gconv_type_d2r:          str = "sage",
    ):
        super().__init__()
        
        # Temporal layer 1
        self.temp1 = HeteroTemporalBlock(ntype_channels_in, ntype_channels_mid, Kt, act_func)

        ##### Spatial layer #####
           
        # Stage 1: property (measurement) -> device
        ## No edge weight, just binary
        if gconv_type_p2d == 'gat':
            self.hetero_conv_1 = HeteroConv({
                ('property', 'measured_by', 'device'): GATConv(
                    in_channels     = (ntype_channels_mid['property'], ntype_channels_mid['device']),
                    out_channels    = ntype_channels_mid['device'] // heads,
                    heads           = heads,
                    concat          = True,
                    bias            = bias,
                    add_self_loops = False,
                )
                }, aggr=aggr
            )
        elif gconv_type_p2d == 'sage':
            self.hetero_conv_1 = HeteroConv({
                ('property', 'measured_by', 'device'): SAGEConv(
                    in_channels     = (ntype_channels_mid['property'], ntype_channels_mid['device']),
                    out_channels    = ntype_channels_mid['device'],
                    bias            = bias
                )
                }, aggr=aggr
            )
        
        # Stage 2: device -> room
        ## No edge weight, just binary
        if gconv_type_d2r == 'gat':
            self.hetero_conv_2 = HeteroConv({
                ('device', 'contained_in', 'room'): GATConv(
                    in_channels     = (ntype_channels_mid['device'], ntype_channels_mid['room']),
                    out_channels    = ntype_channels_mid['room'] // heads,
                    heads           = heads,
                    concat          = True,
                    bias            = bias,
                    add_self_loops = False,
                )
                }, aggr=aggr
            )
        elif gconv_type_d2r == 'sage':
            self.hetero_conv_2 = HeteroConv({
                ('device', 'contained_in', 'room'): SAGEConv(
                    in_channels     = (ntype_channels_mid['device'], ntype_channels_mid['room']),
                    out_channels    = ntype_channels_mid['room'],
                    bias            = bias
                )
                }, aggr=aggr
            )
        
        # Stage 3: broadcast time node and outside node to room nodes
        ## time: no weights
        self.time_proj_room = nn.Linear(
            in_features         = ntype_channels_mid['time'],
            out_features        = ntype_channels_mid['room'], 
            bias                = bias
        )
        ## outside: has outside-room weights!
        self.outside_proj = nn.Linear(
            in_features         = ntype_channels_mid['outside'],
            out_features        = ntype_channels_mid['room'], 
            bias                = bias
        )

        # Stage 4: Spatial relationships between rooms
        ## There are edge weights for these!
        self.hetero_conv_3 = HeteroConv({
            ('room', 'adjacent_horizontal', 'room'): GCNConv(
                in_channels     = ntype_channels_mid['room'],
                out_channels    = ntype_channels_mid['room'],
                bias            = bias,
                add_self_loops  = True
            ),
            ('room', 'adjacent_vertical', 'room'): GCNConv(
                in_channels     = ntype_channels_mid['room'],
                out_channels    = ntype_channels_mid['room'],
                bias            = bias,
                add_self_loops  = True
            ),
            }, aggr=aggr
        )
        
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
    
    @staticmethod
    def _tile_edge_index_over_time(
        edge_index_B: Tensor,
        B: int,
        T: int,
        N_src: int,
        N_dst: int,
        device: torch.device
    ) -> Tensor:
        """
        Given batched-over-B edge_index (2, E_B), tile it over T time steps so indices
        address features flattened over (B * T) graphs per node type.
        """
        # Offsets per time slice for src/dst types
        # shape: (T, 1)
        off_src = (torch.arange(T, device=device) * (B * N_src)).view(T, 1)
        off_dst = (torch.arange(T, device=device) * (B * N_dst)).view(T, 1)

        # Broadcast-add offsets, then reshape back to (2, T*E_B)
        src = edge_index_B[0].unsqueeze(0) + off_src  # (T, E_B)
        dst = edge_index_B[1].unsqueeze(0) + off_dst  # (T, E_B)
        edge_index_BT = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        return edge_index_BT.long()
    
    @staticmethod
    def _flat_bt(x: Tensor) -> Tensor:
        """
        (B, C, T, N) -> (B*T*N, C)
        """
        B, C, T, N = x.shape
        return x.permute(0, 2, 3, 1).reshape(B * T * N, C)
    
    @staticmethod
    def _unflat_bt(x_flat: Tensor, B: int, T: int, N: int, C: int) -> Tensor:
        """
        (B*T*N, C) -> (B, C, T, N)
        """
        return x_flat.view(B, T, N, C).permute(0, 3, 1, 2)
    
    def forward(
        self,
        x_dict: Dict[str, Tensor],          # each: (B, C, T, N)
        edge_index_dict: Dict[tuple, Dict[str, Tensor]]  # {"index": Long[2,E_B], "weight": Optional[...]}
    ) -> Dict[str, Tensor]:
        # ---- Temporal 1 ----
        x_mid = self.temp1(x_dict)  # dict: (B, C_mid, T1, N)
        # Shapes / counts
        B = next(iter(x_mid.values())).shape[0]
        T1 = next(iter(x_mid.values())).shape[2]
        N_room = x_mid['room'].shape[3]
        N_dev  = x_mid['device'].shape[3]
        N_prop = x_mid['property'].shape[3]
        device = x_mid['room'].device

        # -------- Vectorized spatial pipeline --------
        # Flatten (B, T1) into one big "batch" for PyG
        x_flat = {
            'room':     self._flat_bt(x_mid['room']),
            'device':   self._flat_bt(x_mid['device']),
            'property': self._flat_bt(x_mid['property']),
            # time/outside are single-node types; not used in HeteroConv, keep as is.
        }  # shapes: (B*T1*N_type, C_mid_type)
        
        # ---- Stage 1: property -> device (binary) ----
        ei_p2d_B = edge_index_dict[('property', 'measured_by', 'device')]['index'].to(device)
        ei_p2d_BT = self._tile_edge_index_over_time(
            ei_p2d_B, B=B, T=T1, N_src=N_prop, N_dst=N_dev, device=device
        )
        out_p2d = self.hetero_conv_1(
            x_dict={'property': x_flat['property'], 'device': x_flat['device']},
            edge_index_dict={('property', 'measured_by', 'device'): ei_p2d_BT}
        )
        # Merge result
        x_flat['device'] = out_p2d['device']

        # ---- Stage 2: device -> room (binary) ----
        ei_d2r_B = edge_index_dict[('device', 'contained_in', 'room')]['index'].to(device)
        ei_d2r_BT = self._tile_edge_index_over_time(
            ei_d2r_B, B=B, T=T1, N_src=N_dev, N_dst=N_room, device=device
        )
        out_d2r = self.hetero_conv_2(
            x_dict={'device': x_flat['device'], 'room': x_flat['room']},
            edge_index_dict={('device', 'contained_in', 'room'): ei_d2r_BT}
        )
        x_flat['room'] = out_d2r['room']

        # Reshape room/device back to (B, C_mid, T1, N)
        C_room_mid = x_mid['room'].shape[1]
        C_dev_mid  = x_mid['device'].shape[1]
        x_room = self._unflat_bt(x_flat['room'],   B, T1, N_room, C_room_mid)
        x_dev  = self._unflat_bt(x_flat['device'], B, T1, N_dev,  C_dev_mid)

        # ---- Stage 3: broadcast time/outside -> room ----
        # time: (B, C_t, T1, 1) -> project -> (B, C_room, T1, 1) -> broadcast to rooms
        t_proj = self.time_proj_room(x_mid['time'].squeeze(-1).permute(0, 2, 1))  # (B, T1, C_room)
        t_proj = t_proj.permute(0, 2, 1).unsqueeze(-1)                             # (B, C_room, T1, 1)
        t_proj = t_proj.expand(-1, -1, -1, N_room)                                 # (B, C_room, T1, N_room)

        # outside: (B, C_out, T1, 1) -> project -> (B, C_room, T1) then weight per room
        o_proj = self.outside_proj(x_mid['outside'].squeeze(-1).permute(0, 2, 1))  # (B, T1, C_room)
        o_proj = o_proj.permute(0, 2, 1).unsqueeze(-1)                              # (B, C_room, T1, 1)

        # Build per-batch room weights from ('outside','influences','room') edges
        out_edge = edge_index_dict[('outside', 'influences', 'room')]
        dst_idx_B = out_edge['index'][1].to(device)                 # (E_B,)
        w_B = out_edge['weight']
        if w_B is None:
            w_B = torch.ones_like(dst_idx_B, dtype=o_proj.dtype, device=device)
        else:
            w_B = w_B.to(device).view(-1).to(dtype=o_proj.dtype)    # (E_B,)

        # Convert global batched indices -> (batch_id, local_room_id)
        batch_id = dst_idx_B // N_room                               # (E_B,)
        room_loc = dst_idx_B %  N_room                               # (E_B,)

        # Dense weight matrix W_out->room: (B, N_room)
        W = torch.zeros(B, N_room, dtype=o_proj.dtype, device=device)
        W.index_put_((batch_id, room_loc), w_B, accumulate=True)

        # Broadcast outside using weights: (B, 1, 1, N_room)
        W = W.unsqueeze(1).unsqueeze(1)                              # (B, 1, 1, N_room)
        o_proj = o_proj * W                                          # (B, C_room, T1, N_room)

        # Add broadcasts to room
        x_room = x_room + t_proj + o_proj                            # (B, C_room, T1, N_room)

        # ---- Stage 4: room-room diffusion (weighted) ----
        # Flatten room features and tile room-room edges over time
        room_flat = self._flat_bt(x_room)                            # (B*T1*N_room, C_room)

        # Horizontal
        ei_rh_B = edge_index_dict[('room', 'adjacent_horizontal', 'room')]['index'].to(device)
        ew_rh_B = edge_index_dict[('room', 'adjacent_horizontal', 'room')]['weight']
        ei_rh_BT = self._tile_edge_index_over_time(
            ei_rh_B, B=B, T=T1, N_src=N_room, N_dst=N_room, device=device
        )
        ew_rh_BT = None
        if ew_rh_B is not None:
            ew_rh_BT = ew_rh_B.to(device).view(-1).repeat(T1)        # (T1 * E_B,)

        # Vertical
        ei_rv_B = edge_index_dict[('room', 'adjacent_vertical', 'room')]['index'].to(device)
        ew_rv_B = edge_index_dict[('room', 'adjacent_vertical', 'room')]['weight']
        ei_rv_BT = self._tile_edge_index_over_time(
            ei_rv_B, B=B, T=T1, N_src=N_room, N_dst=N_room, device=device
        )
        ew_rv_BT = None
        if ew_rv_B is not None:
            ew_rv_BT = ew_rv_B.to(device).view(-1).repeat(T1)

        # Run GCNs (weighted where provided)
        out_room_dict = self.hetero_conv_3(
            x_dict={'room': room_flat},
            edge_index_dict={
                ('room', 'adjacent_horizontal', 'room'): ei_rh_BT,
                ('room', 'adjacent_vertical', 'room'):   ei_rv_BT,
            },
            edge_weight_dict={
                k: v for k, v in {
                    ('room', 'adjacent_horizontal', 'room'): ew_rh_BT,
                    ('room', 'adjacent_vertical', 'room'):   ew_rv_BT
                }.items() if v is not None
            }
        )
        room_flat = out_room_dict['room']

        # Back to (B, C_mid, T1, N_room)
        x_room = self._unflat_bt(room_flat, B, T1, N_room, C_room_mid)

        # ---- ReLU then Temporal 2 ----
        x_after_spatial = {
            'room':     x_room,
            'device':   x_dev,
            'property': x_mid['property'],
            'outside':  x_mid['outside'],
            'time':     x_mid['time'],
        }
        x_relu = {k: self.relu(v) for k, v in x_after_spatial.items()}

        x_out = self.temp2(x_relu)  # dict: (B, C_out, T2, N)

        # ---- LayerNorm + Dropout ----
        final_x = {}
        for ntype, x in x_out.items():
            # (B, C, T, N) -> (B, T, N, C) -> LN(C) -> back
            x_perm = x.permute(0, 2, 3, 1)
            x_norm = self.norms[ntype](x_perm)
            final_x[ntype] = self.dropout(x_norm).permute(0, 3, 1, 2)

        return final_x