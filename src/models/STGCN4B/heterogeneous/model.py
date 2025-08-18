from typing import Dict, Any, List
import torch
from torch import nn, Tensor

from .layers import HeteroSTBlock
from ..homogeneous.layers import OutputBlock
from ....config.args import build_channel_dicts


class HeterogeneousSTGCN(nn.Module):
    """The final, recommended Heterogeneous STGCN model."""
    def __init__(
        self,
        args:               Any,
        metadata:           tuple,
        all_edges_by_block: List[Dict[tuple, Dict[str, Tensor]]],
        node_feature_dims:  Dict[str, int],
        property_types:     List[str],
        num_devices:        int,
        task_type:          str = "measurement_forecast",
    ):
        super().__init__()
        self.task_type = task_type
        node_types = metadata[0]
        
        # Channel dimensions plan
        ch_mid_plan, ch_out_plan = build_channel_dicts(args, property_types)
        
        # Device (nodes) embeddings
        self.device_embedding = nn.Embedding(
            num_embeddings      = num_devices,
            embedding_dim       = args.device_embed_dim
        )
        # Update node_feature_dims to reflect the embedding dimension
        node_feature_dims = dict(node_feature_dims)
        node_feature_dims['device'] = args.device_embed_dim
        
        # Sanity check if GAT heads
        if args.gconv_type_p2d == "gat":
            dev_ch = ch_mid_plan["device"]
            if dev_ch % args.att_heads != 0:
                raise ValueError("device mid channels must be divisible by att_heads for GAT p->d")
            if args.bidir_p2d:
                for pt in property_types:
                    prop_ch = ch_mid_plan[f"prop_{pt}"]
                    if prop_ch % args.att_heads != 0:
                        raise ValueError(f"prop_{pt} mid channels must be divisible by att_heads for reverse GAT d->p")

        if args.gconv_type_d2r == "gat":
            room_ch = ch_mid_plan["room"]
            dev_ch  = ch_mid_plan["device"]
            if room_ch % args.att_heads != 0:
                raise ValueError(f"room mid channels ({room_ch}) must be divisible by att_heads ({args.att_heads}) for GAT d->r")
            if args.bidir_d2r and (dev_ch % args.att_heads != 0):
                raise ValueError(f"device mid channels ({dev_ch}) must be divisible by att_heads ({args.att_heads}) for reverse GAT r->d")
        
        # ST-Conv Blocks
        self.st_blocks = nn.ModuleList()
        current_dims = dict(node_feature_dims)
        
        for i in range(args.stblock_num):
            mid_dims = ch_mid_plan
            out_dims = ch_out_plan
            self.st_blocks.append(HeteroSTBlock(
                Kt                      = args.Kt,
                ntype_channels_in       = current_dims,
                ntype_channels_mid      = mid_dims,
                ntype_channels_out      = out_dims,
                static_edge_dict        = all_edges_by_block[i],
                property_types          = property_types,
                act_func                = args.act_func,
                bias                    = args.enable_bias,
                droprate                = args.droprate,
                droprate_by_type        = {'time': 0.0, 'outside': 0.0},
                aggr                    = args.aggr_type,
                heads                   = args.att_heads,
                gconv_type_p2d          = args.gconv_type_p2d,
                gconv_type_d2r          = args.gconv_type_d2r,
                bidir_p2d               = args.bidir_p2d,
                bidir_d2r               = args.bidir_d2r,
                gate_mode               = args.gate_mode,
            ))
            current_dims = out_dims
        
        # Output Block for per-node forecasting
        self.Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
        if self.Ko <= 0:
            raise ValueError(f"Invalid architecture: Ko={self.Ko} <= 0.")
        
        out_ch = args.output_channels
        self.output_block = OutputBlock(
            Ko                          = self.Ko,
            last_block_channel          = current_dims['room'],
            channels                    = [out_ch, out_ch],
            end_channel                 = len(args.forecast_horizons),
            n_vertex                    = 1, # We already did the reshaping
            act_func                    = args.act_func,
            bias                        = args.enable_bias,
            droprate                    = 0.0,
        )

    def forward(self, x_pack: Dict[str, Any]):
        """
        Forward pass assumes x_pack["features"] contains tensors of shape (B, C, T, N).
        """
        x_dict = x_pack["features"]
        
        ### Applying the device embeddings at the start of the forward pass ###
        # 1. Get device indices. Shape is (B, 1, T, N_device), dtype=long.
        # The indices are the same across time, so we take them from T=0.
        device_indices = x_dict['device'][:, 0, 0, :].long()  # Shape: (B, N_device)
        
        # 2. Look up the embeddings.
        # Output shape: (B, N_device, D_embedding)
        device_features = self.device_embedding(device_indices)
        
        # 3. Reshape for the ST-Blocks, which expect (B, C, T, N).
        # Permute to (B, D_embedding, N_device)
        device_features = device_features.permute(0, 2, 1)
        
        # 4. Get the history length (T) from another node type and expand.
        # Repeating the static embeddings across the time dimension.
        T = x_dict['room'].shape[2]
        x_dict['device'] = device_features.unsqueeze(2).expand(-1, -1, T, -1)
        # Final shape for 'device' is now (B, D_embedding, T, N_device)
        
        
        for blk in self.st_blocks:
            x_dict = blk(x_dict)
        
        # We only care about 'room' nodes for the final forecast, as that's our target
        x_room = x_dict['room'] # Shape: (B, C, T_out, N_room)
        
        # Reshape for the OutputBlock
        B, C, T, N_room = x_room.shape
        x_room_reshaped = x_room.permute(0, 3, 1, 2).reshape(B * N_room, C, T, 1)
        
        output = self.output_block(x_room_reshaped) # (B*N_room, H, 1, 1)
        
        # Reshape to match target y shape: (B, H, N_room)
        H = output.shape[1]
        output = output.squeeze(-1).squeeze(-1).view(B, N_room, H).permute(0, 2, 1)
        
        return output
    
    def reset_all_parameters(self, seed: int) -> None:
        """
        Re-initialise all learnable parameters and running buffers.

        Parameters
        ----------
        seed : Sets torch's RNG so that each call is deterministic.
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Reset all parameters recursively
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()