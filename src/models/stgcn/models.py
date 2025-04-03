import torch
import torch.nn as nn
from .layers import STConvBlock, OutputBlock

class STGCNChebGraphConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network with Chebyshev graph convolution.
    
    Args:
        args: Model arguments
        blocks: List of channel configurations for each block
        n_vertex: Number of vertices (rooms)
        gso: Graph shift operator (normalized Laplacian)
    """
    def __init__(self, args, blocks, n_vertex, gso):
        super(STGCNChebGraphConv, self).__init__()
        self.blocks = blocks
        self.n_vertex = n_vertex
        self.gso = gso
        
        # Build STGCN blocks (TGTND TGTND TNFF structure)
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                args.act_func, 'cheb_graph_conv', gso, 
                True, args.droprate
            ))
        self.st_blocks = nn.Sequential(*modules)
        
        # Calculate size of remaining temporal kernel for output
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        # Output block
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
        
        # Final pooling layer to convert from [batch, 1, time, vertex] to [batch, 1]
        # For binary classification task
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass of STGCN.
        
        Args:
            x: Input tensor [batch_size, features, time_steps, n_vertex]
            
        Returns:
            x: Output tensor [batch_size, 1] for binary classification
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
        
        # Global pooling to get final classification output
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        return x


class STGCNGraphConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network with simple graph convolution.
    
    Args:
        args: Model arguments
        blocks: List of channel configurations for each block
        n_vertex: Number of vertices (rooms)
        gso: Graph shift operator (normalized adjacency)
    """
    def __init__(self, args, blocks, n_vertex, gso):
        super(STGCNGraphConv, self).__init__()
        self.blocks = blocks
        self.n_vertex = n_vertex
        self.gso = gso
        
        # Build STGCN blocks (TGTND TGTND TNFF structure)
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                args.act_func, 'graph_conv', gso, 
                True, args.droprate
            ))
        self.st_blocks = nn.Sequential(*modules)
        
        # Calculate size of remaining temporal kernel for output
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        # Output block
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
        
        # Final pooling layer to convert from [batch, 1, time, vertex] to [batch, 1]
        # For binary classification task
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass of STGCN.
        
        Args:
            x: Input tensor [batch_size, features, time_steps, n_vertex]
            
        Returns:
            x: Output tensor [batch_size, 1] for binary classification
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
        
        # Global pooling to get final classification output
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        return x
