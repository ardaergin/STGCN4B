import os
import argparse

def add_OfficeGraph_args(parser):
    """
    Parse OfficeGraph arguments.
    
    Related with the code of the following classes:
        - OfficeGraph, with following mixins:
            - OfficeGraphExtractor
        - OfficeGraphBuilder, with following mixins:
            - TemporalBuilderMixin
            - SpatialBuilderMixin
            - HomoGraphBuilderMixin
            - HeteroGraphBuilderMixin
            - TabularBuilderMixin
    """    
    ##############################
    #  Seed
    ##############################
    parser.add_argument('--seed', type=int, default=2658918, 
                      help='Random seed')

    ##############################
    #  Base data arguments
    ##############################
    parser.add_argument('--data_dir', type=str, 
                        default='data', 
                        help='Path to OfficeGraph data directory')
    parser.add_argument('--output_dir', type=str, 
                        default='./output', 
                        help='Directory to save outputs')

    ##############################
    #  Extraction arguments
    ##############################
    parser.add_argument("--floors", type=int, nargs="+",
                        default=[7],
                        help="List of floor numbers to load (default: 7)")
    parser.add_argument("--no-extract",
                        action="store_true",
                        help="Skip automatic entity extraction")
    parser.add_argument("--no-save-pickle",
                        action="store_true",
                        help="Skip saving extracted entities to pickle file")

    ##############################
    #  Builder base arguments
    ##############################
    ##### Base #####
    parser.add_argument('--make_and_save_plots', action='store_true',
                        help='Do the plotting and save the plots (default: False)')
    parser.add_argument('--builder_plots_dir', type=str,
                        default='output/builder',
                        help='Directory to save plot images from the builder')

    ##############################
    #  Temporal Builder
    ##############################
    # Time-related
    parser.add_argument('--start_time', type=str, 
                        default='2022-03-07 00:00:00', 
                        help='Start time for data')
    parser.add_argument('--end_time', type=str, 
                        default='2023-01-29 00:00:00', 
                        help='End time for data')
    parser.add_argument('--interval', type=str, 
                        default="1h",
                        help='Frequency of time buckets as a pandas offset string e.g., ("15min", "30min", "1h", "2h")')
    parser.add_argument('--use_sundays', action='store_true',
                        help='Include Sundays in the time blocks (default: False)')
    
    # Classification arguments
    parser.add_argument('--country_code', type=str,
                        default='NL',
                        help='Country code for work hour classification')

    # Consumption data arguments
    parser.add_argument('--consumption_dir', type=str, 
                        default='data/consumption',
                        help='Directory containing consumption data (for forecasting)')
    
    # Weather data arguments
    parser.add_argument('--weather_csv_path', type=str,
                        default="data/weather/hourly_weather_2022_2023.csv",
                        help='Path to weather data CSV file')
        
    ##############################
    #  Spatial Builder
    ##############################
    # Static room attributes
    parser.add_argument('--static_attr_preset', type=str,
                    choices=['minimal', 'standard', 'all'],
                    default='standard',
                    help='Preset for static room attributes: minimal, standard, or all')

    # Polygon-related arguments
    parser.add_argument('--polygon_type', type=str,
                        choices=['geo', 'doc'],
                        default='doc',
                        help='Type of polygon data to use: geo or doc')
    
    parser.add_argument('--simplify_polygons', action='store_true',
                        dest='simplify_polygons',
                        help='Polygon simplification (off by default)')
    
    parser.add_argument('--simplify_epsilon', type=float,
                        default=0.1,
                        help='Epsilon value for polygon simplification (higher = more simplification)')

    # Adjacency-related arguments
    parser.add_argument('--adjacency_type', type=str,
                        choices=['binary', 'weighted'],
                        default='weighted',
                        help='Type of adjacency: binary or weighted')

    parser.add_argument('--distance_threshold', type=float,
                        default=5.0,
                        help='Distance threshold for room adjacency')

    ##############################
    #  Graph
    ##############################
    parser.add_argument('--data_to_build', type=str, default='graph',
                    choices=['graph', 'tabular'],
                    help='Graph type')
    
    parser.add_argument('--graph_type', type=str, default='homogeneous',
                      choices=['heterogeneous', 'homogeneous'],
                      help='Graph type')
    
    parser.add_argument('--skip_incorporating_weather', 
                        action='store_true',
                        help='Do not add the weather info to the homogeneous graph (default: False)')

    ##############################
    #  GSO (later on in the pipeline)
    ##############################

    parser.add_argument('--gso_mode', type=str, default='dynamic',
                      choices=['static', 'dynamic'], 
                      help='Adjacency matrix type')
    
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


def add_base_modelling_args(parser):
    """Parse common data and training arguments."""
    
    # Task-related arguments
    parser.add_argument('--model', type=str, default='STGCN',
                      choices=['STGCN', 'LightGBM'], 
                      help='Model type')
    parser.add_argument('--task_type', type=str, default='consumption_forecast',
                      choices=['workhour_classification', 'consumption_forecast', 'measurement_forecast'], 
                      help='Task type')
    parser.add_argument('--measurement_type', type=str, default='Temperature',
                      choices=['Temperature', 'Humidity', 'CO2Level'],
                      help='If task_type is "measurement_forecast", then, which measurement type to forecast.')
        
    # Device specification
    parser.add_argument('--device', type=str,
                        default='cpu',
                        help='PyTorch device for tensor operations (cpu, cuda, etc.)')
    parser.add_argument('--enable_cuda', action='store_true', 
                      help='Enable CUDA')
    
    # Common training arguments
    parser.add_argument('--batch_size', type=int, default=144, 
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='Number of epochs')
    
    # Stratified data splitting
    parser.add_argument('--stratum_size', type=int, 
                        default=5, 
                        help='Number of blocks per stratum for the data splitter.')

    # Experimental Setup
    parser.add_argument('--n_outer_splits', type=int,
                         default=2, # For minimal testing
                         help='Number of outer loop train-test splits for nested CV.')
    parser.add_argument('--n_optuna_trials', type=int, 
                        default=2, # For minimal testing
                        help='Number of Optuna trials for HPO.')
    parser.add_argument('--optuna_crash_mode', type=str, default='safe',
                        choices=['fail_fast', 'safe'], 
                        help="weather to add 'study.optimize(..., catch=(Exception,))'.")


def add_STGCN_args(parser):
    """
    STGCN-specific arguments.
    
    STGCN: Spatio-temporal graph convolutional network.
    (Yu et al., 2018)
    """
    # STGCN specific parameters
    parser.add_argument('--n_his', type=int, default=24,
                      help='Number of historical time steps to use')
    parser.add_argument('--n_pred', type=int, default=1,
                        help='the number of time interval for predcition, default as 1')
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
    parser.add_argument('--enable-bias', dest='enable_bias', action='store_true', help='Enable bias in layers.')
    parser.add_argument('--disable-bias', dest='enable_bias', action='store_false', help='Disable bias in layers.')
    parser.set_defaults(enable_bias=True)
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    parser.add_argument('--optimizer', type=str, default='adamw', 
                      choices=['adam', 'adamw', 'sgd'], 
                      help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.0001, 
                      help='Learning rate')


def add_LightGBM_args(parser):
    """
    Tabular‐specific arguments for LightGBM training/evaluation.
    """
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose LightGBM output (shows eval logs)'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=1000,
        help='Number of boosting rounds'
    )
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=50,
        help='Rounds of early stopping'
    )

def parse_args():
    """Parse command-line arguments with model-specific parameters."""
    # Create the parser
    parser = argparse.ArgumentParser(description='OfficeGraph argument parser')
        
    # Add OfficeGraph arguments
    add_OfficeGraph_args(parser)

    # Parsing base modelling arguments
    add_base_modelling_args(parser)
    
    # Add model-specific arguments based on the model type
    temp_args, _ = parser.parse_known_args()
    if temp_args.model == 'STGCN':
        add_STGCN_args(parser)
    elif temp_args.model == 'LightGBM':
        add_LightGBM_args(parser)
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args
