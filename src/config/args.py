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
    parser.add_argument('--task_type', type=str, default='classification',
                      choices=['classification', 'forecasting'], 
                      help='Task type')
    
    # Graph core
    parser.add_argument('--gso_mode', type=str, default='dynamic',
                      choices=['static', 'dynamic'], 
                      help='Adjacency matrix type')
    parser.add_argument('--adjacency_type', type=str,
                        choices=['binary', 'weighted'],
                        default='weighted',
                        help='Type of adjacency that was used: binary or weighted')
    parser.add_argument('--gso_type', type=str, default='rw_norm_adj',
        choices=[
            'sym_norm_adj',  'sym_renorm_adj',  'sym_norm_lap',  'sym_renorm_lap',
            'rw_norm_adj',   'rw_renorm_adj',   'rw_norm_lap',   'rw_renorm_lap',
        ],
        help=(
            "Which Graph-Shift Operator to build:\n"
            "  • sym_norm_adj   : D^{-½} A D^{-½}\n"
            "  • sym_renorm_adj : D^{-½}(A+I)D^{-½}\n"
            "  • sym_norm_lap   : I - D^{-½} A D^{-½}\n"
            "  • sym_renorm_lap : I - D^{-½}(A+I)D^{-½}\n"
            "  • rw_norm_adj    : D^{-1} A\n"
            "  • rw_renorm_adj  : D^{-1}(A+I)\n"
            "  • rw_norm_lap    : I - D^{-1} A\n"
            "  • rw_renorm_lap  : I - D^{-1}(A+I)"
        )
    )

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
    
    parser.add_argument('--batch_size', type=int, default=144, 
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='Number of epochs')
        
    return parser

def add_stgcn_args(parser):
    """
    STGCN-specific arguments.
    
    STGCN: Spatio-temporal graph convolutional network.
    (Yu et al., 2018)
    """
    # STGCN specific parameters
    parser.add_argument('--n_his', type=int, default=24, 
                      help='Number of historical time steps to use')
    parser.add_argument('--n_pred', type=int, default=3, 
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3, 
                      help='Kernel size in temporal convolution')
    parser.add_argument('--Ks', type=int, default=3, 
                      help='Kernel size in graph convolution')
    parser.add_argument('--stblock_num', type=int, default=3, 
                      help='Number of ST-Conv blocks')
    parser.add_argument('--act_func', type=str, default='glu', 
                      choices=['glu', 'gtu', 'relu', 'silu'], 
                      help='Activation function')
    parser.add_argument('--graph_conv_type', type=str, default='graph_conv', 
                      choices=['cheb_graph_conv', 'graph_conv'], 
                      help='Graph convolution type')
    parser.add_argument('--droprate', type=float, default=0.5, 
                      help='Dropout rate')
    parser.add_argument('--step_size', type=int, default=10, 
                      help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.9, 
                      help='Gamma for learning rate scheduler')
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    parser.add_argument('--optimizer', type=str, default='adamw', 
                      choices=['adam', 'adamw', 'sgd'], 
                      help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.0001, 
                      help='Learning rate')

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
