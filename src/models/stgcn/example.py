#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script to run STGCN on OfficeGraph for working hours classification.
This demonstrates a simple use case with default parameters.
"""

import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from .main import main

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Make sure CUDA is available if needed
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        enable_cuda = True
    else:
        logger.info("No GPU available, using CPU")
        enable_cuda = False
    
    # Data directory - adjust to your system
    data_dir = os.environ.get("DATA_DIR", "data/OfficeGraph")
    output_dir = os.environ.get("OUTPUT_DIR", "output/stgcn_officegraph")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Define arguments as a list for the main function
    # You can modify these arguments as needed
    args_list = [
        # Data parameters
        "--data_dir", data_dir,
        "--start_time", "2022-03-01 00:00:00",
        "--end_time", "2023-01-30 00:00:00",
        "--interval_hours", "1",
        "--output_dir", output_dir,
        
        # Model parameters
        "--n_his", "12",  # Use 12-hour history window
        "--Kt", "3",      # Temporal convolution kernel size
        "--Ks", "3",      # Spatial convolution kernel size (Chebyshev order)
        "--stblock_num", "2",  # Number of ST-Conv blocks
        "--act_func", "glu",   # Activation function
        "--graph_conv_type", "cheb_graph_conv",  # Graph convolution type
        "--droprate", "0.5",   # Dropout rate
        
        # Training parameters
        "--seed", "42",
        "--test_ratio", "0.2",
        "--batch_size", "32",
        "--epochs", "100",
        "--lr", "0.001",
        "--weight_decay", "0.0001",
        "--step_size", "10",
        "--gamma", "0.9",
        "--patience", "15",
        "--optimizer", "adamw",
    ]
    
    # Add CUDA flag if available
    if enable_cuda:
        args_list.append("--enable_cuda")
    
    # Convert to argv-style arguments and run main function
    import sys
    sys.argv = ["stgcn_main.py"] + args_list
    
    try:
        logger.info("Starting STGCN training for OfficeGraph working hours classification...")
        main()
        logger.info(f"Training completed successfully! Results saved to {output_dir}")
        
        # Display result image if available
        result_img_path = os.path.join(output_dir, 'stgcn_results.png')
        if os.path.exists(result_img_path):
            logger.info(f"Results visualization saved to {result_img_path}")
            
        # Print metrics
        metrics_path = os.path.join(output_dir, 'stgcn_metrics.txt')
        if os.path.exists(metrics_path):
            logger.info("Final metrics:")
            with open(metrics_path, 'r') as f:
                metrics = f.read()
                logger.info("\n" + metrics)
    
    except Exception as e:
        logger.error(f"Error running STGCN: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
