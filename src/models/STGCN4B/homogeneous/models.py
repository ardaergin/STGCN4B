"""
STGCN variant with support for multiple tasks and dynamic adjacency structure.

This script is a modified version of the original models.py from
https://github.com/hazdzz/stgcn/blob/main/model/models.py

Main enhancements:
- Supports different task types via `task_type` parameter:
    • 'default': full spatio-temporal prediction (original STGCN)
    • 'measurement_forecast': predicts time series values for each node (e.g., temperature, CO₂)
    • 'consumption_forecast': predicts a single aggregated target (e.g., building-wide consumption)
    • 'workhour_classification': classifies each time window (e.g., work hour vs. non-work hour)
- Modular design with output pooling and optional classifiers for consumption/classification tasks
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
from typing import List, Literal
import torch
import torch.nn as nn

from .layers import STConvBlock, OutputBlock


class HomogeneousSTGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network supporting
    • Chebyshev or GCN 1-hop graph convolutions
    • Four task types: default, measurement_forecast,
      consumption_forecast, workhour_classification

    Parameters
    ----------
    args          : argparse.Namespace that provides model arguments 
                    Kt, Ks, n_his, act_func, droprate, enable_bias
    blocks        : List of channel configurations for each block
    n_vertex      : Number of graph nodes
    gso           : Graph shift operator, can be a list for dynamic progression in each block.
    conv_type     : "gcn" or "cheb"
    task_type     : "default", 'workhour_classification', 'consumption_forecast', or 'measurement_forecast'
    num_classes   : Number of classes (only used for classification task)
    """
    def __init__(
        self, 
        args:       argparse.Namespace, 
        blocks:     List[List[int]], 
        n_vertex:   int, 
        gso:        torch.Tensor | List[torch.Tensor], 
        *,
        conv_type:      Literal["gcn", "cheb"] = "gcn",
        task_type:      str = "measurement_forecast",
        num_classes:    int | None = None
    ):
        super().__init__()
        
        if conv_type not in {"gcn", "cheb"}:
            raise ValueError("conv_type must be 'gcn' or 'cheb'")
        self.task_type = task_type
        self.blocks = blocks
        self.n_vertex = n_vertex
        
        # Handle per-block GSOs (allow single tensor or iterable)
        if not isinstance(gso, (list, tuple)):
            gso = [gso] * (len(blocks) - 3)
        if len(gso) != len(blocks) - 3:
            raise ValueError(f"Need {len(blocks)-3} GSOs for {len(blocks)-3} blocks, got {len(gso)}")
        self.gso = gso
        
        ########## Spatio-Temporal blocks ##########
        # (T -> S -> T - > N -> D)
        conv_op_name = "cheb_graph_conv" if conv_type == "cheb" else "graph_conv"
        st_layers = []
        for l, gso_l in enumerate(self.gso):
            st_layers.append(
                STConvBlock(
                    Kt=args.Kt, 
                    Ks=args.Ks, 
                    n_vertex=n_vertex,
                    last_block_channel=blocks[l][-1],
                    channels=blocks[l+1],
                    act_func=args.act_func,
                    graph_conv_type=conv_op_name,
                    gso=gso_l,
                    bias=args.enable_bias,
                    droprate=args.droprate,
                )
            )
        self.st_blocks = nn.Sequential(*st_layers)
        
        ########## Output block ##########
        # Calculate size of remaining temporal kernel for output
        self.Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)

        if self.Ko > 1:
            self.output = OutputBlock(
                Ko=self.Ko,                    # remaining temporal size
                last_block_channel=blocks[-3][-1],  # last ST block's out-channels
                channels=blocks[-2],
                end_channel=blocks[-1][0],
                n_vertex=n_vertex,
                act_func=args.act_func,
                bias=args.enable_bias,
                droprate=args.droprate,
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
        if task_type in {"consumption_forecast", "workhour_classification"}:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        if task_type == "workhour_classification":
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
            default              -> (B, C_out, T_out, V)
            measurement_forecast -> (B, C_out, V)
            consumption_forecast -> (B, C_out)
            workhour_classif.    -> (B, num_classes)
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

        elif self.task_type == 'consumption_forecast':
            x = self.global_pool(x)                # [B,1,1,1]
            return x.squeeze(-1).squeeze(-1)       # [B, n_pred]

        elif self.task_type == 'workhour_classification':
            # Global pooling to get final classification output
            x = self.global_pool(x).squeeze(-1).squeeze(-1)
            # Apply classifier for multi-class if needed
            if hasattr(self, 'classifier'):
                x = self.classifier(x)
            return x
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")