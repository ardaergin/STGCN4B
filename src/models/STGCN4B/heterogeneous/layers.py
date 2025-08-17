from typing import Mapping, Dict, List
import torch
from torch import nn, Tensor
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
            static_edge_dict:       Dict[tuple, Dict[str, Tensor]],
            property_types:         List[str],
            act_func:               str = "glu",
            bias:                   bool = True,
            droprate:               float = 0.0,
            droprate_by_type:       Dict[str,float] | None = None,
            aggr:                   str = "sum",
            heads:                  int = 4,
            gconv_type_p2d:         str = "sage",
            gconv_type_d2r:         str = "sage",
            bidir_p2d:              bool = False,
            bidir_d2r:              bool = False,
            gate_mode:              str = "scalar",
    ):
        super().__init__()
        self.bidir_p2d = bidir_p2d
        self.bidir_d2r = bidir_d2r
        self.static_edge_dict = static_edge_dict
        self.property_types = property_types
        
        # Temporal layer 1
        self.temp1 = HeteroTemporalBlock(ntype_channels_in, ntype_channels_mid, Kt, act_func)

        ##### Spatial layer #####
           
        # Stage 1: property (measurement) -> device
        ## No edge weight, just binary
        convs_1: Dict[tuple, nn.Module] = {}
        
        for prop_type in property_types:
            prop_node_type = f'prop_{prop_type}'
            
            if gconv_type_p2d == 'gat':
                # Forward: property -> device
                convs_1[(prop_node_type, 'measured_by', 'device')] = GATConv(
                    in_channels     = (ntype_channels_mid[prop_node_type], ntype_channels_mid['device']),
                    out_channels    = ntype_channels_mid['device'] // heads,
                    heads           = heads,
                    concat          = True,
                    bias            = bias,
                    add_self_loops  = False,
                )
                if self.bidir_p2d:
                    # Reverse: device -> property
                    convs_1[('device', f'measures_{prop_type}', prop_node_type)] = GATConv(
                        in_channels     = (ntype_channels_mid['device'], ntype_channels_mid[prop_node_type]),
                        out_channels    = ntype_channels_mid[prop_node_type] // heads,
                        heads           = heads,
                        concat          = True,
                        bias            = bias,
                        add_self_loops  = False,
                    )
            elif gconv_type_p2d == 'sage':
                # Forward: property -> device
                convs_1[(prop_node_type, 'measured_by', 'device')] = SAGEConv(
                    in_channels     = (ntype_channels_mid[prop_node_type], ntype_channels_mid['device']),
                    out_channels    = ntype_channels_mid['device'],
                    bias            = bias
                )
                if self.bidir_p2d:
                    # Reverse: device -> property
                    convs_1[('device', f'measures_{prop_type}', prop_node_type)] = SAGEConv(
                        in_channels     = (ntype_channels_mid['device'], ntype_channels_mid[prop_node_type]),
                        out_channels    = ntype_channels_mid[prop_node_type],
                        bias            = bias
                    )
        self.hetero_conv_1 = HeteroConv(convs_1, aggr=aggr)
                        
        # Stage 2: device -> room
        ## No edge weight, just binary
        convs_2: Dict[tuple, nn.Module] = {}
        if gconv_type_d2r == 'gat':
            convs_2[('device', 'contained_in', 'room')] = GATConv(
                in_channels     = (ntype_channels_mid['device'], ntype_channels_mid['room']),
                out_channels    = ntype_channels_mid['room'] // heads,
                heads           = heads,
                concat          = True,
                bias            = bias,
                add_self_loops  = False,
            )
            if self.bidir_d2r:
                convs_2[('room', 'contains', 'device')] = GATConv(
                    in_channels     = (ntype_channels_mid['room'], ntype_channels_mid['device']),
                    out_channels    = ntype_channels_mid['device'] // heads,
                    heads           = heads,
                    concat          = True,
                    bias            = bias,
                    add_self_loops  = False,
                )
        elif gconv_type_d2r == 'sage':
            convs_2[('device', 'contained_in', 'room')] = SAGEConv(
                in_channels     = (ntype_channels_mid['device'], ntype_channels_mid['room']),
                out_channels    = ntype_channels_mid['room'],
                bias            = bias
            )
            if self.bidir_d2r:
                convs_2[('room', 'contains', 'device')] = SAGEConv(
                    in_channels     = (ntype_channels_mid['room'], ntype_channels_mid['device']),
                    out_channels    = ntype_channels_mid['device'],
                    bias            = bias
                )
        self.hetero_conv_2 = HeteroConv(convs_2, aggr=aggr)
        
        # Stage 3: broadcast time node and outside node to room nodes
        ## time: no weights
        self.time_proj_scale = nn.Linear(
            in_features         = ntype_channels_mid['time'],
            out_features        = ntype_channels_mid['room'], 
            bias                = bias
        )
        self.time_proj_shift = nn.Linear(
            in_features         = ntype_channels_mid['time'],
            out_features        = ntype_channels_mid['room'], 
            bias                = bias
        )
        ## outside: has outside-room weights!
        self.outside_proj_scale = nn.Linear(
            in_features         = ntype_channels_mid['outside'], 
            out_features        = ntype_channels_mid['room'], 
            bias                = bias
        )
        self.outside_proj_shift = nn.Linear(
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
                add_self_loops  = False,
                normalize       = False,
            ),
            ('room', 'adjacent_vertical', 'room'): GCNConv(
                in_channels     = ntype_channels_mid['room'],
                out_channels    = ntype_channels_mid['room'],
                bias            = bias,
                add_self_loops  = False,
                normalize       = False,
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
        self.norms = nn.ModuleDict({ntype: nn.LayerNorm(C_out) for ntype, C_out in ntype_channels_out.items()})
        self.dropouts = nn.ModuleDict({
            ntype: nn.Dropout((droprate_by_type or {}).get(ntype, droprate))
            for ntype in ntype_channels_out.keys()
        })
        
        # Learnable gates
        self.gate_mode = gate_mode
        
        def make_gate(size, init=-2.0):  # scalar or per-channel vector
            if gate_mode == "scalar":
                return nn.Parameter(torch.tensor(init))          # ~0.12 after sigmoid
            elif gate_mode == "channel":
                return nn.Parameter(torch.full((size,), init))   # one gate per channel
            else:
                raise ValueError("gate_mode must be 'scalar' or 'channel'")
        
        # Gates for p<->d block
        self.g_p2d_dev  = make_gate(ntype_channels_mid['device'], init=+2.0)
        
        if self.bidir_p2d:
            self.g_d2p_prop = nn.ParameterDict({
                f'prop_{pt}': make_gate(ntype_channels_mid[f'prop_{pt}'], init=-2.0)
                for pt in property_types
            })
        
        # Gates for d<->r block
        self.g_d2r_room = make_gate(ntype_channels_mid['room'], init=+2.0)
        if self.bidir_d2r:
            self.g_r2d_dev  = make_gate(ntype_channels_mid['device'], init=-2.0)
        
        # broadcast gates (slightly on by default)
        self.g_time2room    = make_gate(ntype_channels_mid['room'], init=2.0)
        self.g_outside2room = make_gate(ntype_channels_mid['room'], init=-2.0)

    @staticmethod
    def _blend_with_gate(theta: torch.Tensor, x_old: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
        """
        Blends tensors with a learnable gate. Handles both 2D (..., C) and 
        4D (B, C, T, N) tensors.
        """
        alpha = torch.sigmoid(theta)
        if alpha.dim() == 0:  # Scalar gate
            return (1 - alpha) * x_old + alpha * x_new

        # Per-channel gate
        if x_new.dim() == 2:
            # Reshape for (..., C) tensor
            alpha_reshaped = alpha.view(1, -1)
        elif x_new.dim() == 4:
            # Reshape for (B, C, T, N) tensor
            alpha_reshaped = alpha.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimension {x_new.dim()} in _blend_with_gate")
            
        return (1 - alpha_reshaped) * x_old + alpha_reshaped * x_new
    
    @staticmethod
    def _tile_edge_index_over_bt(
        edge_index: Tensor,  # (2, E) single-graph indices
        B: int,
        T: int,
        N_src: int,
        N_dst: int,
        device: torch.device,
    ) -> Tensor:
        # indices layout after _flat_bt: idx = t*(B*N) + b*N + n
        t = torch.arange(T, device=device).view(T,1).repeat(1,B).reshape(-1)  # [T*B]
        b = torch.arange(B, device=device).view(1,B).repeat(T,1).reshape(-1)  # [T*B]
        off_src = t * (B*N_src) + b * N_src
        off_dst = t * (B*N_dst) + b * N_dst
        src = edge_index[0].unsqueeze(0) + off_src.view(-1,1)   # [T*B,E]
        dst = edge_index[1].unsqueeze(0) + off_dst.view(-1,1)   # [T*B,E]
        return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0).long()
    
    @staticmethod
    def _flat_bt(x: Tensor) -> Tensor:
        """
        (B, C, T, N) -> (T*B*N, C)  time-major flatten
        """
        B, C, T, N = x.shape
        return x.permute(2, 0, 3, 1).reshape(T * B * N, C)
    
    @staticmethod
    def _unflat_bt(x_flat: Tensor, B: int, T: int, N: int, C: int) -> Tensor:
        """
        (T*B*N, C) -> (B, C, T, N)
        """
        return x_flat.view(T, B, N, C).permute(1, 3, 0, 2)
    
    def forward(
        self,
        x_dict: Dict[str, Tensor],          # each: (B, C, T, N)
    ) -> Dict[str, Tensor]:
        # ---- Temporal 1 ----
        x_mid = self.temp1(x_dict)  # dict: (B, C_mid, T1, N)
        # Shapes / counts
        B = next(iter(x_mid.values())).shape[0]
        T1 = next(iter(x_mid.values())).shape[2]
        N_room = x_mid['room'].shape[3]
        N_dev  = x_mid['device'].shape[3]
        device = x_mid['room'].device
        edge_index_dict = self.static_edge_dict
        
        # Count nodes for each property type
        N_prop_by_type = {}
        for prop_type in self.property_types:
            prop_node_type = f'prop_{prop_type}'
            if prop_node_type in x_mid:
                N_prop_by_type[prop_node_type] = x_mid[prop_node_type].shape[3]
        
        # -------- Vectorized spatial pipeline --------
        # Flatten (B, T1) into one big "batch" for PyG
        # Shapes: (B*T1*N_type, C_mid_type)
        x_flat = {
            'room':     self._flat_bt(x_mid['room']),
            'device':   self._flat_bt(x_mid['device']),
        }
        for prop_type in self.property_types:
            prop_node_type = f'prop_{prop_type}'
            if prop_node_type in x_mid:
                x_flat[prop_node_type] = self._flat_bt(x_mid[prop_node_type])
        

        # ---- Stage 1: property -> device (binary) ----
        edge_idx_1 = {}
        
        # Build edge indices for each property type
        for prop_type in self.property_types:
            prop_node_type = f'prop_{prop_type}'
            
            # Skip if this property type doesn't exist in the graph
            if prop_node_type not in N_prop_by_type:
                continue
                
            N_prop = N_prop_by_type[prop_node_type]
            
            # Forward edges: property -> device
            edge_key_forward = (prop_node_type, 'measured_by', 'device')
            if edge_key_forward in edge_index_dict:
                ei_p2d_B = edge_index_dict[edge_key_forward]['index'].to(device)
                ei_p2d_BT = self._tile_edge_index_over_bt(
                    ei_p2d_B, B=B, T=T1, N_src=N_prop, N_dst=N_dev, device=device
                )
                edge_idx_1[edge_key_forward] = ei_p2d_BT
            
            # Reverse edges: device -> property (if bidirectional)
            if self.bidir_p2d:
                edge_key_reverse = ('device', f'measures_{prop_type}', prop_node_type)
                if edge_key_reverse in edge_index_dict:
                    ei_d2p_B = edge_index_dict[edge_key_reverse]['index'].to(device)
                    ei_d2p_BT = self._tile_edge_index_over_bt(
                        ei_d2p_B, B=B, T=T1, N_src=N_dev, N_dst=N_prop, device=device
                    )
                    edge_idx_1[edge_key_reverse] = ei_d2p_BT

        # Build input dict for hetero_conv_1
        x_dict_stage1 = {'device': x_flat['device']}
        for prop_node_type in N_prop_by_type.keys():
            x_dict_stage1[prop_node_type] = x_flat[prop_node_type]

        out_bi_1 = self.hetero_conv_1(
            x_dict=x_dict_stage1,
            edge_index_dict=edge_idx_1
        )
        
        # Always update device (aggregated from all property types)
        x_flat['device'] = self._blend_with_gate(self.g_p2d_dev, x_flat['device'], out_bi_1['device'])
        
        # Update each property type if reverse relation is enabled
        if self.bidir_p2d:
            for prop_type in self.property_types:
                prop_node_type = f'prop_{prop_type}'
                if prop_node_type in out_bi_1:
                    x_flat[prop_node_type] = self._blend_with_gate(
                        self.g_d2p_prop[prop_node_type], 
                        x_flat[prop_node_type], 
                        out_bi_1[prop_node_type]
                    )
        

        # ---- Stage 2: device -> room (binary) ----
        ei_d2r_B  = edge_index_dict[('device', 'contained_in', 'room')]['index'].to(device)
        ei_d2r_BT = self._tile_edge_index_over_bt(ei_d2r_B, B=B, T=T1, N_src=N_dev, N_dst=N_room, device=device)
        
        edge_idx_2 = {('device', 'contained_in', 'room'): ei_d2r_BT}
        if self.bidir_d2r:
            ei_r2d_B  = edge_index_dict[('room', 'contains', 'device')]['index'].to(device)
            ei_r2d_BT = self._tile_edge_index_over_bt(ei_r2d_B, B=B, T=T1, N_src=N_room, N_dst=N_dev, device=device)
            edge_idx_2[('room', 'contains', 'device')] = ei_r2d_BT
        
        out_bi_2 = self.hetero_conv_2(
            x_dict={'device': x_flat['device'], 'room': x_flat['room']},
            edge_index_dict=edge_idx_2
        )
        # Always update room (forward d->r)
        x_flat['room'] = self._blend_with_gate(self.g_d2r_room, x_flat['room'], out_bi_2['room'])
        # Update device only if reverse relation is enabled
        if self.bidir_d2r and 'device' in out_bi_2:
            x_flat['device'] = self._blend_with_gate(self.g_r2d_dev, x_flat['device'], out_bi_2['device'])
        
        # Reshape room/device back to (B, C_mid, T1, N)
        C_room_mid = x_mid['room'].shape[1]
        C_dev_mid  = x_mid['device'].shape[1]
        x_room = self._unflat_bt(x_flat['room'],   B, T1, N_room, C_room_mid)
        x_dev  = self._unflat_bt(x_flat['device'], B, T1, N_dev,  C_dev_mid)
        x_props = {}
        for prop_type in self.property_types:
            prop_node_type = f'prop_{prop_type}'
            if prop_node_type in N_prop_by_type:
                C_prop_mid = x_mid[prop_node_type].shape[1]
                N_prop = N_prop_by_type[prop_node_type]
                x_props[prop_node_type] = self._unflat_bt(
                    x_flat[prop_node_type], B, T1, N_prop, C_prop_mid
                )
        

        # ---- Stage 3: broadcast time/outside -> room ----
        # --- 3a. Apply Time FiLM ---
        time_features_flat = x_mid['time'].squeeze(-1).permute(0, 2, 1)  # (B, T1, C_time)
        
        gamma_time = self.time_proj_scale(time_features_flat).permute(0, 2, 1).unsqueeze(-1).expand_as(x_room)
        beta_time = self.time_proj_shift(time_features_flat).permute(0, 2, 1).unsqueeze(-1).expand_as(x_room)

        film_effect_time = (gamma_time * x_room) + beta_time
        x_room_after_time = self._blend_with_gate(self.g_time2room, x_room, film_effect_time)

        # --- 3b. Apply Outside FiLM (using the output from the time step) ---
        outside_features_flat = x_mid['outside'].squeeze(-1).permute(0, 2, 1) # (B, T1, C_outside)

        gamma_outside = self.outside_proj_scale(outside_features_flat).permute(0, 2, 1).unsqueeze(-1)
        beta_outside = self.outside_proj_shift(outside_features_flat).permute(0, 2, 1).unsqueeze(-1)
        
        # Get static weights and expand them to the right shape for broadcasting
        dst = self.static_edge_dict[('outside','influences','room')]['index'][1].to(device)
        w = self.static_edge_dict[('outside','influences','room')]['weight']
        w = (torch.ones_like(dst, dtype=gamma_outside.dtype, device=device) if w is None
            else w.to(device).view(-1).to(gamma_outside.dtype))
        
        W_room = torch.zeros(N_room, dtype=gamma_outside.dtype, device=device)
        W_room.index_add_(0, dst, w)
        W = W_room.view(1, 1, 1, N_room).expand(B, -1, T1, -1) # Shape: (B, 1, T1, N_room)

        # Use static weights to scale the FiLM parameters
        gamma_outside_weighted = gamma_outside * W
        beta_outside_weighted = beta_outside * W
        
        # Apply the weighted FiLM transformation to the time-modulated features
        film_effect_outside = (gamma_outside_weighted * x_room_after_time) + beta_outside_weighted
        x_room_after_outside = self._blend_with_gate(self.g_outside2room, x_room_after_time, film_effect_outside)
        
        # Final 'room' tensor to be used in the next stage
        x_room = x_room_after_outside
        
        # ---- Stage 4: room-room diffusion (weighted) ----
        # Flatten room features and tile room-room edges over time
        room_flat = self._flat_bt(x_room)                            # (B*T1*N_room, C_room)

        # Horizontal
        ei_rh_B = edge_index_dict[('room', 'adjacent_horizontal', 'room')]['index'].to(device)
        ew_rh_B = edge_index_dict[('room', 'adjacent_horizontal', 'room')]['weight']
        ei_rh_BT = self._tile_edge_index_over_bt(
            ei_rh_B, B=B, T=T1, N_src=N_room, N_dst=N_room, device=device
        )
        ew_rh_BT = None

        # Vertical
        ei_rv_B = edge_index_dict[('room', 'adjacent_vertical', 'room')]['index'].to(device)
        ew_rv_B = edge_index_dict[('room', 'adjacent_vertical', 'room')]['weight']
        ei_rv_BT = self._tile_edge_index_over_bt(
            ei_rv_B, B=B, T=T1, N_src=N_room, N_dst=N_room, device=device
        )
        ew_rv_BT = None

        # Room to room weights
        ew_rh_BT = ew_rh_B.to(device).view(-1).repeat(T1 * B) if ew_rh_B is not None else None
        ew_rv_BT = ew_rv_B.to(device).view(-1).repeat(T1 * B) if ew_rv_B is not None else None

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
            'outside':  x_mid['outside'],
            'time':     x_mid['time'],
        }
        for prop_node_type, x_prop in x_props.items():
            x_after_spatial[prop_node_type] = x_prop
        
        x_relu = {k: self.relu(v) for k, v in x_after_spatial.items()}

        x_out = self.temp2(x_relu)  # dict: (B, C_out, T2, N)

        # ---- LayerNorm + Dropout ----
        final_x = {}
        for ntype, x in x_out.items():
            # (B, C, T, N) -> (B, T, N, C) -> LN(C) -> back
            x_perm = x.permute(0, 2, 3, 1)
            x_norm = self.norms[ntype](x_perm)
            x_do   = self.dropouts[ntype](x_norm) # type-specific dropout
            final_x[ntype] = x_do.permute(0, 3, 1, 2)

        return final_x