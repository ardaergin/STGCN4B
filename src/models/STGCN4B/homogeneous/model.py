"""
STGCN variant with support for multiple tasks and dynamic adjacency structure.

This script is a modified version of the original models.py from
https://github.com/hazdzz/stgcn/blob/main/model/models.py

Main enhancements:
- Supports different task types via `task_type` parameter:
    • 'default': full spatio-temporal prediction (original STGCN)
    • 'measurement_forecast': predicts time series values for each node (e.g., temperature, CO₂)
    • 'single_value_forecast': predicts a single aggregated target (e.g., building-wide consumption)
    • 'single_value_classification': classifies each time window (e.g., work hour vs. non-work hour)
- Modular design with output pooling and optional classifiers for single_value/classification tasks
- Dynamic support for per-block graph shift operators (GSOs), including Chebyshev and GCN graph convolutions

Model variants:
- `STGCNChebGraphConv`: alias of `HomogeneousSTGCN` with `conv_type='cheb'`,
                        uses Chebyshev polynomial graph convolutions.
- `STGCNGraphConv`: alias of `HomogeneousSTGCN` with `conv_type='gcn'`,
                    uses gcn 1-hop graph convolutions

Note:
- Input shape: (batch_size, input_channels, time_steps, n_nodes)
- Output shape depends on the selected task_type
"""

import argparse
from typing import List, Literal, Any
import torch
import torch.nn as nn

from .layers import STConvBlock, OutputBlock


class HomogeneousSTGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network supporting
    • Chebyshev or GCN 1-hop graph convolutions
    • Four task types: 
        1. default, 
        2. measurement_forecast,
        3. single_value_forecast,
        4. single_value_classification

    Parameters
    ----------
    args          : argparse.Namespace that provides model arguments 
                    Kt, Ks, n_his, act_func, droprate, enable_bias
    blocks        : List of channel configurations for each block
    n_vertex      : Number of graph nodes
    gso           : Graph shift operator, can be a list for dynamic progression in each block.
    conv_type     : "gcn" or "cheb"
    task_type     : "default", "measurement_forecast", "single_value_forecast", or "single_value_classification"
    num_classes   : Number of classes (only used for classification)
    """
    def __init__(
        self, 
        args:           argparse.Namespace, 
        n_vertex:       int,
        n_features:     int,
        task_type:      str,
        gso:            torch.Tensor | List[torch.Tensor] | None    = None,
        conv_type:      Literal["gcn", "cheb", "none"]              = "gcn",
        num_classes:    int | None                                  = None
    ):
        super().__init__()
        
        if conv_type not in {"gcn", "cheb", "none"}:
            raise ValueError("conv_type must be 'gcn' or 'cheb' or 'none'")
        if conv_type == "none" and not args.drop_spatial_layer:
            raise ValueError("conv_type='none' requires drop_spatial_layer=True")
        
        self.task_type = task_type
        self.n_vertex = n_vertex
        self.n_features = n_features
        
        # Define the STGCN architecture
        blocks, Ko = self.define_stgcn_architecture(args, n_features)
        self.blocks = blocks
        self.Ko = Ko

        # Handle per-block GSOs (allow single tensor or iterable)
        if args.drop_spatial_layer:
            self.gso = [None] * (len(blocks) - 3)
        else:
            if not isinstance(gso, (list, tuple)):
                gso = [gso] * (len(blocks) - 3)
            if len(gso) != len(blocks) - 3:
                raise ValueError(f"Need {len(blocks)-3} GSOs for {len(blocks)-3} blocks, got {len(gso)}")
            self.gso = gso
        
        ########## Spatio-Temporal blocks ##########
        if args.drop_spatial_layer:     conv_op_name = None
        else:                           conv_op_name = "cheb_graph_conv" if conv_type == "cheb" else "graph_conv"
        
        st_layers = []
        for l, gso_l in enumerate(self.gso):
            st_layers.append(
                STConvBlock(
                    Kt                  = args.Kt,
                    Ks                  = args.Ks,
                    n_vertex            = n_vertex,
                    last_block_channel  = blocks[l][-1],
                    channels            = blocks[l+1],
                    act_func            = args.act_func,
                    graph_conv_type     = conv_op_name,
                    gso                 = gso_l,
                    bias                = args.enable_bias,
                    droprate            = args.droprate,
                    drop_spatial_layer  = args.drop_spatial_layer if hasattr(args, 'drop_spatial_layer') else False,
                )
            )
        self.st_blocks = nn.Sequential(*st_layers)
        
        ########## Output block ##########
        if self.Ko > 1:
            self.output = OutputBlock(
                Ko                      = self.Ko,                    # remaining temporal size
                last_block_channel      = blocks[-3][-1],  # last ST block's out-channels
                channels                = blocks[-2],
                end_channel             = blocks[-1][0],
                n_vertex                = n_vertex,
                act_func                = args.act_func,
                bias                    = args.enable_bias,
                droprate                = args.droprate,
            )
        elif self.Ko == 0:
            # If no more temporal dimension left, use only fully connected layers
            self.fc1 = nn.Linear(
                blocks[-3][-1], 
                blocks[-2][0],
                bias=args.enable_bias)
            self.fc2 = nn.Linear(
                blocks[-2][0], 
                blocks[-1][0],
                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        else: 
            raise ValueError(
                f"Remaining temporal size Ko became negative (Ko={self.Ko}). "
                "This means the chosen hyper-parameters consume more timesteps than "
                f"the input provides. Check n_his={args.n_his}, Kt={args.Kt} and the "
                f"number of ST blocks ({len(blocks)-3}); Ko must be ≥ 0."
            )
        
        ########## Extras for pooling / classification ##########
        if task_type in {"single_value_forecast", "single_value_classification"}:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        if task_type == "single_value_classification":
            if num_classes is None:
                num_classes = 1   # binary by default
            if num_classes > 1:
                self.classifier = nn.Linear(
                    blocks[-1][0], 
                    num_classes, 
                    bias=args.enable_bias)
    
    def forward(self, x):
        """
        Forward pass of HomogeneousSTGCN.
                    
        Shapes
        ------
        Input : (batch_size, features, time_steps, n_vertex)
        Output:
            default                         -> (B, C_out, T_out, V)
            measurement_forecast            -> (B, C_out, V)
            single_value_forecast           -> (B, C_out)
            single_value_classification     -> (B, num_classes)
        """
        # Pass through ST blocks
        x = self.st_blocks(x)
        
        # Pass through output block
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        # Different output processing based on task type
        if self.task_type == 'default': # the original STGCN architecture
            return x                    # [B,C,T,V]

        elif self.task_type=='measurement_forecast':
            # x is [B, C_out=n_pred, T_out=1, N]
            # drop the temporal axis (dim=2), keep the channel axis of length n_pred
            x = x.squeeze(2)            # (B, C_out=n_pred, N)
            return x                    # which aligns perfectly with y: (B, n_pred, N)

        elif self.task_type == 'single_value_forecast':
            x = self.global_pool(x)                # [B,1,1,1]
            return x.squeeze(-1).squeeze(-1)       # [B, n_pred]

        elif self.task_type == 'single_value_classification':
            # Global pooling to get final classification output
            x = self.global_pool(x).squeeze(-1).squeeze(-1)
            # Apply classifier for multi-class if needed
            if hasattr(self, 'classifier'):
                x = self.classifier(x)
            return x
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
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

    @staticmethod
    def define_stgcn_architecture(args: Any, n_features: int) -> List[List[int]]:
        """Defines the STGCN model's layer structure based on hyperparameters."""
        blocks = []
        
        # Input features
        blocks.append([n_features])
        
        # Intermediate ST-Conv blocks
        for st_block in range(args.stblock_num):
            blocks.append([args.st_main_channels, args.st_bottleneck_channels, args.st_main_channels])
        
        # Output blocks
        # > Calculating the size of remaining temporal kernel for output
        Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
        if Ko > 0:
            blocks.append([args.output_channels, args.output_channels])
        elif Ko == 0:
            blocks.append([args.output_channels])
        else:
            raise ValueError(f"Invalid architecture: Ko={Ko}. Adjust n_his, stblock_num, or Kt.")
        
        # Final output layer
        blocks.append([len(args.forecast_horizons)])
        
        return blocks, Ko