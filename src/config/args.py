#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

def parse_base_args(parser=None):
    """Parse common data and training arguments."""
    if parser is None:
        parser = argparse.ArgumentParser(description='OfficeGraph Analysis')
    
    # Task-related arguments
    parser.add_argument('--model', type=str, default='stgcn',
                      choices=['stgcn', 'astgcn'], 
                      help='Model type')
    parser.add_argument('--task', type=str, default='classification',
                      choices=['classification', 'forecasting'], 
                      help='Task type')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', 
                      help='Path to OfficeGraph data directory')
    parser.add_argument('--start_time', type=str, default='2022-03-01 00:00:00', 
                      help='Start time for analysis')
    parser.add_argument('--end_time', type=str, default='2023-01-30 00:00:00', 
                      help='End time for analysis')
    parser.add_argument('--interval_hours', type=int, default=1, 
                      help='Size of time buckets in hours')
    parser.add_argument('--output_dir', type=str, default='./output', 
                      help='Directory to save results')
    parser.add_argument('--include_sundays', action='store_true',
                      help='Include Sundays in the time blocks (default: False)')
    
    # Common training parameters
    parser.add_argument('--enable_cuda', action='store_true', 
                      help='Enable CUDA')
    parser.add_argument('--seed', type=int, default=42, 
                      help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                      help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, 
                      help='Patience for early stopping')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                      choices=['adam', 'adamw', 'sgd'], 
                      help='Optimizer type')
    
    return parser

def add_stgcn_args(parser):
    """
    STGCN-specific arguments.
    
    STGCN: Spatio-temporal graph convolutional network.
    (Yu et al., 2018)
    """
    # STGCN specific parameters
    parser.add_argument('--n_his', type=int, default=12, 
                      help='Number of historical time steps to use')
    parser.add_argument('--Kt', type=int, default=3, 
                      help='Kernel size in temporal convolution')
    parser.add_argument('--Ks', type=int, default=3, 
                      help='Kernel size in graph convolution')
    parser.add_argument('--stblock_num', type=int, default=2, 
                      help='Number of ST-Conv blocks')
    parser.add_argument('--act_func', type=str, default='glu', 
                      choices=['glu', 'gtu', 'relu', 'silu'], 
                      help='Activation function')
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', 
                      choices=['cheb_graph_conv', 'graph_conv'], 
                      help='Graph convolution type')
    parser.add_argument('--droprate', type=float, default=0.5, 
                      help='Dropout rate')
    parser.add_argument('--step_size', type=int, default=10, 
                      help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.9, 
                      help='Gamma for learning rate scheduler')
    
    return parser

def add_astgcn_args(parser):
    """
    ASTGCN-specific arguments.
    ASTGCN: Attention-based spatial-temporal graph convolutional network.
    (Guo et al., 2019)
    """
    # Not implemented yet.
    pass

def parse_args():
    """Parse command-line arguments with model-specific parameters."""
    # Create the parser
    parser = argparse.ArgumentParser(description='OfficeGraph Analysis')
        
    # Add all base arguments
    parser = parse_base_args(parser)
    
    # Parse just to get the model type
    temp_args, _ = parser.parse_known_args()
    
    # Add model-specific arguments based on the model type
    if temp_args.model == 'stgcn':
        parser = add_stgcn_args(parser)
    elif temp_args.model == 'astgcn':
        parser = add_astgcn_args(parser)
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args
