import torch
from torch import nn
from .layers import HeteroSTBlock
from typing import Dict, Any
from ..homogeneous.layers import OutputBlock

class HeterogeneousSTGCN(nn.Module):
    """The final, recommended Heterogeneous STGCN model."""
    def __init__(
        self,
        args: Any,
        metadata: tuple,
        node_feature_dims: Dict[str, int],
        task_type: str = "measurement_forecast",
    ):
        super().__init__()
        self.task_type = task_type
        node_types = metadata[0]
        
        # Channel dimensions plan
        st_main = args.st_main_channels
        out_ch = args.output_channels
        
        # ST-Conv Blocks
        self.st_blocks = nn.ModuleList()
        current_dims = node_feature_dims
        for _ in range(args.stblock_num):
            mid_dims = {nt: st_main for nt in node_types}
            out_dims = {nt: st_main for nt in node_types}
            self.st_blocks.append(HeteroSTBlock(
                Kt                      = args.Kt,
                ntype_channels_in       = current_dims,
                ntype_channels_mid      = mid_dims,
                ntype_channels_out      = out_dims,
                metadata                = metadata,
                act_func                = args.act_func,
                droprate                = args.droprate,
                bias                    = args.enable_bias
            ))
            current_dims = out_dims
        
        # Output Block for per-node forecasting
        self.Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
        if self.Ko <= 0:
            raise ValueError(f"Invalid architecture: Ko={self.Ko} <= 0.")

        self.output_block = OutputBlock(
            Ko                          = self.Ko,
            last_block_channel          = current_dims['room'],
            channels                    = [out_ch, out_ch],
            end_channel                 = len(args.forecast_horizons),
            n_vertex                    = 1, # We already did the reshaping
            act_func                    = args.act_func,
            bias                        = args.enable_bias,
            droprate                    = args.droprate,
        )

    def forward(self, x_pack: Dict[str, Any]):
        """
        Forward pass assumes x_pack["features"] contains tensors of shape (B, C, T, N).
        """
        x_dict = x_pack["features"]
        edge_index_dict = x_pack["edges"]
        
        for blk in self.st_blocks:
            x_dict = blk(x_dict, edge_index_dict)

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