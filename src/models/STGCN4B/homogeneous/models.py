"""
STGCN variant with support for multiple tasks and dynamic adjacency structure.

This script is a modified version of the original models.py from
https://github.com/hazdzz/stgcn/blob/main/model/models.py

Main enhancements:
- Supports three task types via `task_type` parameter:
    • 'measurement_forecast': predicts time series values for each node (e.g., temperature, CO₂)
    • 'consumption_forecast': predicts a single aggregated target (e.g., building-wide consumption)
    • 'workhour_classification': classifies each time window (e.g., work hour vs non-work hour)
- Modular design with output pooling and optional classifiers for consumption/classification tasks
- Dynamic support for per-block graph shift operators (GSOs), including Chebyshev and standard convolutions

Model variants:
- `STGCNChebGraphConv`: uses Chebyshev polynomial graph convolutions
- `STGCNGraphConv`: uses simple 1-hop graph convolutions

Usage:
    # Full spatio-temporal prediction (original form)
    model = STGCNGraphConv(..., task_type='default_STGCN')

    # Forecasting measurements for each node (multi-output)
    model = STGCNGraphConv(..., task_type='measurement_forecast')

    # Forecasting a single value (e.g., building-level consumption)
    model = STGCNGraphConv(..., task_type='consumption_forecast')

    # Classifying a window (e.g., workhour detection)
    model = STGCNGraphConv(..., task_type='workhour_classification', num_classes=N)

Note:
- Input shape: (batch_size, input_channels, time_steps, n_nodes)
- Output shape depends on the selected task_type
"""

import torch
import torch.nn as nn
from .layers import STConvBlock, OutputBlock


class STGCNChebGraphConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network with Chebyshev graph convolution.
    Supports both forecasting and classification tasks.
    
    Args:
        args: Model arguments
        blocks: List of channel configurations for each block
        n_vertex: Number of vertices (rooms)
        gso: Graph shift operator (normalized Laplacian)
        task_type: 'workhour_classification', 'consumption_forecast', or 'measurement_forecast'
        num_classes: Number of classes (only used for classification task)
    """
    def __init__(self, args, blocks, n_vertex, gso, task_type='measurement_forecast', num_classes=None):
        super(STGCNChebGraphConv, self).__init__()
        self.blocks = blocks
        self.n_vertex = n_vertex
        # gso(s) can be a single GSO or a list of GSOs, one per block
        if not isinstance(gso, (list, tuple)):
            gso = [gso] * (len(blocks) - 3)
        assert len(gso) == len(blocks) - 3, \
            f"Need {len(blocks)-3} GSOs for {len(blocks)-3} blocks, got {len(gso)}"
        self.gso = gso
        self.task_type = task_type
        
        # Build STGCN blocks (TGTND TGTND TNFF structure)
        modules = []
        for l, gso_l in enumerate(self.gso):
            modules.append(STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                args.act_func, 'cheb_graph_conv', gso_l, 
                True, args.droprate
            ))
        self.st_blocks = nn.Sequential(*modules)
        
        # Calculate size of remaining temporal kernel for output
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        # Output block for forecasting task
        if self.Ko > 1:
            self.output = OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                n_vertex, args.act_func, True, args.droprate
            )
        elif self.Ko == 0:
            # If no more temporal dimension left, use only fully connected layers
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=True)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=True)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        if task_type in ('consumption_forecast', 'workhour_classification'):
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Additional layers for classification task
        if task_type == 'workhour_classification':
            if num_classes is None:
                num_classes = 1
            if num_classes > 1:
                self.classifier = nn.Linear(blocks[-1][0], num_classes)
    
    def forward(self, x):
        """
        Forward pass of STGCN.
        
        Args:
            x: Input tensor [batch_size, features, time_steps, n_vertex]
            
        Returns:
            For forecasting: Output tensor [batch_size, pred_len, time_steps, n_vertex]
            For classification: Output tensor [batch_size, num_classes]
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
        if self.task_type == 'default_STGCN': # the original
            return x                    # [B,C,T,V]

        elif self.task_type=='measurement_forecast':
            # x is [B,C,T,V], i.e.: (batch_size, 1, n_pred, n_vertex)
            # We can drop that singleton channel
            return x.squeeze(1)   # → (batch_size, n_pred, n_vertex)

        elif self.task_type == 'consumption_forecast':
            x = self.global_pool(x)                # [B,1,1,1]
            return x.view(x.size(0))               # [B]

        elif self.task_type == 'workhour_classification':
            # Global pooling to get final classification output
            x = self.global_pool(x).squeeze(-1).squeeze(-1)
            # Apply classifier for multi-class if needed
            if hasattr(self, 'classifier'):
                x = self.classifier(x)
            return x
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")


class STGCNGraphConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network with simple graph convolution.
    
    Args:
        args: Model arguments
        blocks: List of channel configurations for each block
        n_vertex: Number of vertices (rooms)
        gso: Graph shift operator (normalized adjacency)
        task_type: 'workhour_classification', 'consumption_forecast', or 'measurement_forecast'
        num_classes: Number of classes (only used for classification task)
    """
    def __init__(self, args, blocks, n_vertex, gso, task_type='forecasting', num_classes=None):
        super(STGCNGraphConv, self).__init__()
        self.blocks = blocks
        self.n_vertex = n_vertex
        # gso can be a single GSO or a list of GSOs, one per block
        if not isinstance(gso, (list, tuple)):
            gso = [gso] * (len(blocks) - 3)
        assert len(gso) == len(blocks) - 3, \
            f"Need {len(blocks)-3} GSOs for {len(blocks)-3} blocks, got {len(gso)}"
        self.gso = gso
        self.task_type = task_type
        
        # Build STGCN blocks (TGTND TGTND TNFF structure)
        modules = []
        for l, gso_l in enumerate(self.gso):
            modules.append(STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                args.act_func, 'graph_conv', gso_l, 
                True, args.droprate
            ))
        self.st_blocks = nn.Sequential(*modules)
        
        # Calculate size of remaining temporal kernel for output
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        # Output block for prediction task
        if self.Ko > 1:
            self.output = OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                n_vertex, args.act_func, True, args.droprate
            )
        elif self.Ko == 0:
            # If no more temporal dimension left, use only fully connected layers
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=True)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=True)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

        # Additional layers for classification task
        if task_type == 'workhour_classification':
            if num_classes is None:
                num_classes = 1  # Default to binary classification
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            # Optional additional classification layer for multi-class
            if num_classes > 1:
                self.classifier = nn.Linear(blocks[-1][0], num_classes)

        if task_type == 'consumption_forecast':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass of STGCN.
        
        Args:
            x: Input tensor [batch_size, features, time_steps, n_vertex]
            
        Returns:
            For forecasting: Output tensor [batch_size, pred_len, time_steps, n_vertex]
            For classification: Output tensor [batch_size, num_classes]
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
        if self.task_type == 'default_STGCN': # the original
            return x                    # [B,C,T,V]

        elif self.task_type=='measurement_forecast':
            # x is [B,C,T,V], i.e.: (batch_size, 1, n_pred, n_vertex)
            # We can drop that singleton channel
            return x.squeeze(1)   # → (batch_size, n_pred, n_vertex)

        elif self.task_type == 'consumption_forecast':
            x = self.global_pool(x)                # [B,1,1,1]
            return x.view(x.size(0))               # [B]
    
        elif self.task_type == 'workhour_classification':
            # Global pooling to get final classification output
            x = self.global_pool(x).squeeze(-1).squeeze(-1)
            # Apply classifier for multi-class if needed
            if hasattr(self, 'classifier'):
                x = self.classifier(x)
            return x
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
