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
    parser.add_argument('--seed', type=int, 
                        default=2658918, 
                        help='Random seed')

    ##############################
    #  Base data arguments
    ##############################
    parser.add_argument('--data_dir', type=str, 
                        default='data', 
                        help='Path to OfficeGraph data directory')
    parser.add_argument('--output_dir', type=str, 
                        default='output', 
                        help='Directory to save outputs')
    parser.add_argument('--processed_data_dir', type=str, 
                        default='data/processed', 
                        help='Path to OfficeGraph data directory')

    ##############################
    #  Extraction arguments
    ##############################
    parser.add_argument("--floors", type=int, nargs="+",
                        default=[1,2,3,4,5,6,7],
                        help="List of floor numbers to load (default: all)")
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
                        default='output/visualizations',
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
                        default="30min",
                        help='Frequency of time buckets as a pandas offset string e.g., ("15min", "30min", "1h", "2h")')
    parser.add_argument('--use_sundays', action='store_true',
                        help='Include Sundays in the time blocks (default: False)')
    
    # Workhour arguments
    parser.add_argument('--country_code', type=str,
                        default='NL',
                        help='Country code for work hour classification')
    parser.add_argument("--workhour_start", type=int,
                        default=8,
                        choices=range(0, 24),
                        help="Start of working hours (0-23). Default: 8.")
    parser.add_argument("--workhour_end", type=int,
                        default=18,
                        choices=range(0, 24),
                        help="End of working hours (0-23). Default: 18.")
    
    # Consumption data arguments
    parser.add_argument('--consumption_dir', type=str, 
                        default='data/consumption',
                        help='Directory containing consumption data (for forecasting)')
    
    # Weather data arguments
    parser.add_argument('--weather_csv_path', type=str,
                        default="data/weather/hourly_weather_2022_2023.csv",
                        help='Path to weather data CSV file')
    parser.add_argument("--weather_mode", type=str,
                        default="feature",
                        choices=["feature", "node"],
                        help="How to incorporate weather data: 'feature' (add to all rooms) or 'node' (add as separate room node)")
    
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
    #  Forecast Arguments
    ##############################
    # Feature engineering
    parser.add_argument('--lags', type=int, nargs='+',
                        default=[1, 2, 3, 24],
                        help='A space-separated list of lag features to create (e.g., --lags 1 2 8 24).')
    parser.add_argument('--windows', type=int, nargs='+',
                        default=[3, 6, 12, 24],
                        help='A space-separated list of window sizes for moving averages (e.g., --windows 3 6 12).')
    parser.add_argument('--shift_amount', type=int,
                        default=1,
                        help='The shift amount for moving average features. Default is 1.')
    
    # Target engineering
    parser.add_argument('--forecast_horizons', type=int, nargs='+',
                        default=[1],
                        help='A space-separated list of forecast horizons in hours (e.g., --forecast_horizons 1 24). Default is [1].')
    parser.add_argument('--prediction_type', type=str,
                        choices=['absolute', 'delta'],
                        default="delta",
                        help='Whether to predict the actual target value ("absolute") or the change from the current value ("delta"). Default is "delta".')
    parser.add_argument('--forecast_type', type=str,
                        choices=['point', 'range'],
                        default='point',
                        help="The type of forecast to generate. 'point' predicts a single value, 'range' is for aggregated values. Default is 'point'.")
    parser.add_argument('--aggregation_type', type=str,
                        choices=['mean', 'sum'],
                        default='mean',
                        help="The aggregation to apply for 'range' forecasts. Can be 'mean' or 'sum'. Default is 'mean'.")
    parser.add_argument('--min_periods_ratio', type=float,
                        default=0.5,
                        help="The minimum ratio of non-null data points required in a window to compute a value. Default is 0.5.")
    
    # Workhour masking
    parser.add_argument("--do_not_mask_workhours", action="store_false",
                        dest="mask_workhours",
                        help="Disable work-hour masking. By default, masking is enabled."
    )

    parser.add_argument("--hetero_prop_features", type=str,
        nargs="+",
        default=["mean", "std", "max", "min", "count", "has_measurement"],
        help=(
            "Subset of property-level temporal features to use in heterogeneous graphs. "
            "Default is the full set: mean, std, max, min, count, has_measurement."
        ),
    )
    parser.add_argument("--hetero_mode", type=str,
        default="full",
        choices=["full", "minimal"],
        help="Mode for heterogeneous graph construction. Specifies the feature set to use."
        "Minimal if 'mean' + 'has_measurement', full if all features."
    )

    parser.add_argument(
        "--clear_high_counts",
        action="store_true",
        help="If set, clears all rows where count > 1 and drops std/max/min/count columns (keeps only mean + has_measurement)."
    )

    parser.add_argument(
        "--drop_high_missingness",
        action="store_true",
        help=(
            "Drop device–property pairs with high missingness (default threshold 75%). "
            "If not set, only logs missingness statistics."
        ),
    )
    
    parser.add_argument(
        "--missingness_threshold",
        type=float,
        default=0.15, # seperation of R5 with others
        help="Threshold for maximum allowed missingness fraction per device–property (default: 0.75).",
    )


def add_base_modelling_args(parser):
    """Parse common data and training arguments."""
    
    parser.add_argument(
        '--log_level', type=str,
        default='INFO',
        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'],
        help='Set the logging level for the experiment.'
    )

    # Run mode for experimenter
    parser.add_argument('--run_mode', type=str, default='single_run',
                        choices=['single_run', 'experiment'],
                        help='Run mode for the experiment runners.')
    
    # Model arguments
    parser.add_argument('--model_family', type=str, default='graph',
                        choices=['graph', 'tabular'],
                        help='Graph type')
    parser.add_argument('--graph_type', type=str, default='homogeneous',
                        choices=['heterogeneous', 'homogeneous'],
                        help='Graph type')
    parser.add_argument('--model', type=str, default='STGCN',
                        choices=['STGCN', 'LightGBM', 'naive'], 
                        help='Model type')
    
    # Task-related arguments
    parser.add_argument('--task_type', type=str, default='measurement_forecast',
                        choices=['consumption_forecast', 'measurement_forecast'], 
                        help='Task type')
    parser.add_argument('--measurement_variable', type=str, default='Temperature',
                        choices=['Temperature', 'Humidity', 'CO2Level'],
                        help='If task_type is "measurement_forecast", then, which measurement type to forecast.')
    parser.add_argument('--measurement_variable_stat', type=str, default='mean',
                        choices=['mean', 'max', 'min'],
                        help='Which statistic to use for creating targets for the measurement variable.')
    
    # Device specification
    parser.add_argument('--device', type=str,
                        default='cpu',
                        help='PyTorch device for tensor operations (cpu, cuda, etc.)')
    parser.add_argument('--enable_cuda', action='store_true', 
                        help='Enable CUDA')
    
    # Common training arguments
    parser.add_argument('--batch_size', type=int, 
                        default=64, 
                        help='Batch size')
    
    parser.add_argument('--do_not_drop_last_train_batch', action='store_false', 
                        dest='drop_last_batch',
                        default=True,
                        help='If specified, the last batch of the training set will NOT be dropped. '
                            'By default, the last batch is dropped.')
    
    parser.add_argument('--epochs', type=int, 
                        default=50, 
                        help='Number of epochs')
    parser.add_argument('--final_epoch_multiplier', type=float, 
                        default=1.2, 
                        help='Multiply the optimal epochs with this factor for the final training.')
    
    # Stratified data splitting
    parser.add_argument('--stratum_size', type=int, 
                        default=5, 
                        help='Number of blocks per stratum for the data splitter.')

    # Normalization
    parser.add_argument('--normalization_method', type=str,
                        default='median',
                        choices=['mean', 'median'],
                        help='Normalization method')
    parser.add_argument('--skip_normalization_for', nargs='*', 
                        default=['_sin', '_cos', 'wc_', 
                                'has_measurement',
                                'hasWindows', 'has_multiple_windows', 
                                'window_direction_sin', 'window_direction_cos', 
                                'isProperRoom', 'norm_areas_minmax',
                                'norm_area_minmax', 'norm_area_prop',
                                # Time features
                                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_workhour',
                                # Heterogeneous:
                                'measures_', 'is_type_', # device nodes
                                'property_type_' # property nodes
                                ],
                        help='List of (sub-)strings for feature names that should NOT be normalized.')

    parser.add_argument('--norm_clip_value', type=float, 
                        default=10.0,
                        help='Value to clip normalized features to. E.g., 10.0 means features will be in [-10, 10].')

    parser.add_argument('--norm_eps_scale', type=float, 
                        default=1e-6,
                        help='Small epsilon value to use as a minimum scale (denominator) to prevent division by zero.')
    
    parser.add_argument('--norm_log1p_feature_keys', nargs='*', 
                        default=["count", "std"],
                        help='List of substrings (e.g., "count", "std") to identify features that should be log1p transformed before normalization.')
    
    parser.add_argument('--norm_alpha_wide_spread', type=float, 
                        default=0.3,
                        help='Mixture coefficient for the wide-spread percentile range in the robust scaler. A value between 0 and 1.')
    
    
    # Features to drop
    parser.add_argument('--features_to_drop', type=str, nargs='*',
                        default=[],
                        help='A space-separated list of feature columns to drop from the dataframe before training (e.g., --features-to-drop col_a col_b).')
    
    ########## Experimental Setup ##########
    parser.add_argument('--experiment_id', type=int, 
                        default=0,
                        help='The ID of the outer loop train-test split to run (for parallel execution).')
    
    # Optuna general args
    parser.add_argument('--n_optuna_trials', type=int, 
                        default=50,
                        help='Number of Optuna trials for HPO.')
    
    parser.add_argument('--n_jobs_in_hpo', type=int, 
                        default=1,
                        help='Number of parallel threads to use for training (-1 to use all available cores)')
    
    parser.add_argument('--optuna_crash_mode', type=str, default='safe',
                        choices=['fail_fast', 'safe'], 
                        help="whether to add 'study.optimize(..., catch=(Exception,))'.")
    
    # Arguments for configuring Optuna's MedianPruner
    parser.add_argument('--n_startup_trials', type=int, default=5,
                        help='Number of trials to complete before pruning is activated. '
                             'These first trials will always run to completion.')
    parser.add_argument('--n_warmup_steps', type=int, default=10,
                        help="Number of steps (epochs in this case) to complete within a trial "
                             "before it can be pruned. This prevents pruning on initial noisy performance.")
    parser.add_argument('--interval_steps', type=int, default=2,
                        help='Interval (in steps/epochs) at which to check for pruning possibilities '
                             'after the warmup period is over.')
    
    # Pruning slow models
    parser.add_argument("--max_epoch_duration", type=int,
                        default=90,
                        help="Maximum duration for a single epoch in seconds, for pruning slow trials. "
                            "Pruning starts after the first epoch, to accomodate for compiling the model.")


def add_STGCN_args(parser):
    """
    STGCN-specific arguments.
    
    STGCN: Spatio-temporal graph convolutional network.
    (Yu et al., 2018)
    """
    parser.add_argument('--drop_spatial_layer', action='store_true',
                        help='Whether to turn off the spatial layer (Default: False).')
    
    # Compilation arguments
    parser.add_argument('--compile_model', action='store_true',
                        dest='compile_model',
                        default=False,
                        help='Enable model compilation. (Default: compilation is disabled).')
    
    parser.add_argument('--compile_mode', type=str,
                        default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help="The compilation mode for torch.compile. (default: 'default')")
    parser.add_argument('--compile_fullgraph', action='store_true',
                        dest='compile_fullgraph',
                        default=False,
                        help="If set, uses fullgraph=True in torch.compile. (default: False)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dynamic_compile', dest='dynamic_compile_mode', action='store_const', const=True)
    group.add_argument('--static_compile', dest='dynamic_compile_mode', action='store_const', const=False)
    parser.set_defaults(dynamic_compile_mode=None)

    # Data loader args
    parser.add_argument('--num_dataloader_workers', type=int,
                        default=4,
                        help='Number of worker processes for data loading.')
    parser.add_argument('--padding_strategy', type=str,
                        default='zero',
                        choices=['zero', 'replication'],
                        help='Padding strategy for history: "zero" for zero-padding (default), "replication" for replication-padding.')
    # Autocast
    parser.add_argument('--amp', action='store_true', 
                        help='Enable autocast mixed precision')
    parser.add_argument('--amp_dtype', choices=['bf16', 'fp16'], 
                        default='bf16',
                        help='Autocast dtype (bf16 recommended on H100)')
    parser.add_argument('--tf32', action='store_true', 
                        help='Enable TF32 matmul speedups for FP32 ops')
    
    parser.add_argument('--max_grad_norm', type=float, 
                        default=2.0, 
                        help='Max norm for gradient clipping. Set to 0 to disable.')
    
    ########## GSO ##########

    parser.add_argument('--gso_mode', type=str, default='dynamic',
                        choices=['static', 'dynamic'], 
                        help='Adjacency matrix type')
    
    parser.add_argument('--gso_type', type=str, default='rw_renorm_adj',
        choices=[
            'sym_norm_adj',  'sym_renorm_adj',  'sym_norm_lap',  'sym_renorm_lap',
            'rw_norm_adj',   'rw_renorm_adj',   'rw_norm_lap',   'rw_renorm_lap',
            'col_renorm_adj', 'no_norm_only_self_loop'
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
            "  • rw_renorm_lap  : I - D^{-1}(A+I)"))
    
    parser.add_argument('--transpose_gso', action='store_true',
                        help='If set, applies transpose to the GSO.')
    
    ########## STGCN parameters ##########
    parser.add_argument('--n_his', type=int, 
                        default=24,
                        help='Number of historical time steps to use')
    
    parser.add_argument('--Kt', type=int, 
                        default=3, 
                        help='Kernel size in temporal convolution')
    parser.add_argument('--Ks', type=int, 
                        default=3, 
                        help='Kernel size in graph convolution')
    
    parser.add_argument('--stblock_num', type=int, 
                        default=3, 
                        help='Number of ST-Conv blocks')
    
    parser.add_argument('--graph_conv_type', type=str, 
                        default='gcn', 
                        choices=['gcn', 'cheb'], 
                        help='Graph convolution type')
    
    parser.add_argument('--act_func', type=str, 
                        default='glu', 
                        choices=['glu', 'gtu', 'relu', 'silu'], 
                        help='Activation function')

    # Channel size arguments
    parser.add_argument('--st_main_channels', type=int, 
                        default=64,
                        help='Number of main channels in the ST-Conv blocks (e.g., C in T-G(C,B)-T(B,C)).')
    parser.add_argument('--st_bottleneck_channels', type=int, 
                        default=16,
                        help='Number of bottleneck channels in the graph convolution layer within ST-Conv blocks.')
    parser.add_argument('--output_channels', type=int, 
                        default=128,
                        help='Number of channels in the final output block.')

    # Early stopping
    parser.add_argument('--es_patience', type=int, 
                        default=10, 
                        help='early stopping patience')
    parser.add_argument('--es_delta', type=float, 
                        default=0.01, 
                        help='early stopping delta (default: 0.01)')

    # Common training parameters
    parser.add_argument('--lr', type=float, 
                        default=0.0005, 
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, 
                        default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], 
                        help='Optimizer type')
    parser.add_argument('--step_size', type=int, 
                        default=10, 
                        help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float,
                        default=0.9, 
                        help='Gamma for learning rate scheduler')

    parser.add_argument('--droprate', type=float, 
                        default=0.2, 
                        help='Dropout rate')
    parser.add_argument('--weight_decay_rate', type=float, 
                        default=0.001, 
                        help='weight decay (L2 penalty)')
    
    parser.add_argument('--enable_bias', dest='enable_bias', 
                        action='store_true', 
                        help='Enable bias in layers.')
    parser.add_argument('--disable_bias', dest='enable_bias', 
                        action='store_false', 
                        help='Disable bias in layers.')
    parser.set_defaults(enable_bias=True)

    parser.add_argument('--aggr_type', type=str, 
                        default='sum', 
                        choices=['sum', 'mean', 'max'], 
                        help='Aggregation type for hetero')
    
    parser.add_argument('--att_heads', type=int, 
                        default=4, 
                        help='Number of attention heads for GATConv')
    
    parser.add_argument('--gconv_type_p2d', type=str, 
                        default='sage', 
                        choices=['sage', 'gat'], 
                        help='Graph convolution type for property to device')
    parser.add_argument('--gconv_type_d2r', type=str, 
                        default='sage', 
                        choices=['sage', 'gat'], 
                        help='Graph convolution type for device to room')
    parser.add_argument('--bidir_p2d', action='store_true',
                        dest='bidir_p2d',
                        help='Enable bidirectional property to device graph convolution')
    parser.add_argument('--bidir_d2r', action='store_true',
                        dest='bidir_d2r',
                        help='Enable bidirectional device to room graph convolution')
    parser.add_argument('--gate_mode', type=str,
                        default='scalar',
                        choices=['scalar', 'channel'],
                        help='Gate mode for blending features.')

    # Per-type channels. Two ways to pass them:
    # (A) compact "room=64,device=32,property=32,time=8,outside=8"
    parser.add_argument("--ch-mid", type=str, default="")   # optional
    parser.add_argument("--ch-out", type=str, default="")   # optional

    # (B) individual flags (used if present; otherwise fallback to compact or defaults)
    parser.add_argument("--ch-room-mid", type=int, default=64)
    parser.add_argument("--ch-device-mid", type=int, default=32)
    parser.add_argument("--ch-property-mid", type=int, default=32)
    parser.add_argument("--ch-time-mid", type=int, default=8)
    parser.add_argument("--ch-outside-mid", type=int, default=8)

    parser.add_argument("--ch-room-out", type=int, default=64)
    parser.add_argument("--ch-device-out", type=int, default=32)
    parser.add_argument("--ch-property-out", type=int, default=32)
    parser.add_argument("--ch-time-out", type=int, default=8)
    parser.add_argument("--ch-outside-out", type=int, default=8)

def _parse_kv_ints(s: str) -> dict:
    """
    Parse 'room=64,device=32,property=32,time=8,outside=8' -> dict
    """
    out = {}
    if not s:
        return out
    for kv in s.split(","):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out

def build_channel_dicts(args, property_types: list[str]) -> tuple[dict, dict]:
    """
    Returns (mid_dict, out_dict) for node types you have.
    Compact string values override individual flags. 
    Missing keys fall back to the flag defaults.
    """
    compact_mid = _parse_kv_ints(args.ch_mid)
    compact_out = _parse_kv_ints(args.ch_out)

    # Base widths (single "property" width reused for all prop_* types)
    room_mid     = compact_mid.get("room",     args.ch_room_mid)
    device_mid   = compact_mid.get("device",   args.ch_device_mid)
    prop_mid_w   = compact_mid.get("property", args.ch_property_mid)
    time_mid     = compact_mid.get("time",     args.ch_time_mid)
    outside_mid  = compact_mid.get("outside",  args.ch_outside_mid)

    room_out     = compact_out.get("room",     args.ch_room_out)
    device_out   = compact_out.get("device",   args.ch_device_out)
    prop_out_w   = compact_out.get("property", args.ch_property_out)
    time_out     = compact_out.get("time",     args.ch_time_out)
    outside_out  = compact_out.get("outside",  args.ch_outside_out)

    mid = {
        "room":    room_mid,
        "device":  device_mid,
        "time":    time_mid,
        "outside": outside_mid,
        **{f"prop_{pt}": prop_mid_w for pt in property_types},
    }
    out = {
        "room":    room_out,
        "device":  device_out,
        "time":    time_out,
        "outside": outside_out,
        **{f"prop_{pt}": prop_out_w for pt in property_types},
    }
    return mid, out


def add_LightGBM_args(parser):
    """
    Tabular‐specific arguments for LightGBM training/evaluation.
    """    
    # Core LightGBM parameters
    parser.add_argument('--objective', type=str, 
                        default='regression',
                        help='Learning task objective (e.g., "regression", "binary")')
    parser.add_argument('--metric', type=str, 
                        default='mae',
                        help='Metric to be evaluated on the validation set (e.g., "mae", "rmse", "auc")')
    parser.add_argument('--boosting_type', type=str, 
                        default='gbdt',
                        help='Type of boosting algorithm to use')
    parser.add_argument('--n_estimators', type=int, 
                        default=1000,
                        help='Number of boosting rounds')
    
    # Verbosity
    parser.add_argument('--verbosity', type=int, 
                        default=1,
                        help='Controls the level of LightGBM verbosity')
    
    # Early stopping
    parser.add_argument('--early_stopping_rounds', type=int, 
                        default=10,
                        help='Activates early stopping. The model will train until the validation score stops improving.')

    # Tree structure
    parser.add_argument('--num_leaves', type=int, 
                        default=31,
                        help='Maximum number of leaves in one tree')
    parser.add_argument('--max_depth', type=int, 
                        default=-1,
                        help='Maximum tree depth for base learners, -1 means no limit')
    parser.add_argument('--min_child_samples', type=int, 
                        default=20,
                        help='Minimum number of data needed in a child (leaf)')
    parser.add_argument('--min_child_weight', type=float, 
                        default=1e-3,
                        help='Minimum sum of instance weight needed in a child (leaf)')
    parser.add_argument('--min_split_gain', type=float, 
                        default=0.0,
                        help='Minimum loss reduction required to make a further partition on a leaf node')

    # Regularization
    parser.add_argument('--lambda_l1', type=float, 
                        default=0.0,
                        help='L1 regularization term on weights')
    parser.add_argument('--lambda_l2', type=float, 
                        default=0.0,
                        help='L2 regularization term on weights')

    # Sampling & feature selection
    parser.add_argument('--feature_fraction', type=float, 
                        default=0.9,
                        help='Fraction of features to be considered for each tree')
    parser.add_argument('--bagging_fraction', type=float, 
                        default=0.8,
                        help='Fraction of data to be used for each iteration (tree)')
    parser.add_argument('--bagging_freq', type=int, 
                        default=5,
                        help='Frequency for bagging')

    # Learning control
    parser.add_argument('--learning_rate', type=float, 
                        default=0.05,
                        help='Boosting learning rate')
    parser.add_argument('--boost_from_average', dest='boost_from_average', 
                        action='store_true',
                        help='Starts training from the average of the target values')
    parser.add_argument('--no_boost_from_average', dest='boost_from_average', 
                        action='store_false',
                        help='Starts training from the initial base score (usually 0)')
    parser.set_defaults(boost_from_average=True)
    
    # Optional for Classification
    parser.add_argument('--is_unbalance', dest='is_unbalance', 
                        action='store_true',
                        help='Set to true if training data is unbalanced (for binary classification)')
    parser.add_argument('--is_not_unbalance', dest='is_unbalance', 
                        action='store_false',
                        help='Set to false if training data is balanced')
    parser.set_defaults(is_unbalance=True)



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