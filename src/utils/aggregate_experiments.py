import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import optuna
import glob
import argparse
import logging
import math
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def aggregate_and_report(results_dir: str):
    """
    Aggregates results from multiple parallel experiment runs.
    
    Args:
        results_dir (str): The base directory containing all the experiment subdirectories (e.g., 'experiment_0', 'experiment_1', ...).
    """
    logger.info(f"Starting aggregation for results in: {results_dir}")

    # --- 1. Find all individual result files ---
    test_files = glob.glob(os.path.join(results_dir, "experiment_*", "results_test.csv"))
    cv_files = glob.glob(os.path.join(results_dir, "experiment_*", "results_CV.csv"))

    if not test_files:
        logger.error("No 'results_test.csv' files found. Aborting.")
        return
    if not cv_files:
        logger.warning("No 'results_CV.csv' files found. CV analysis will be skipped.")

    logger.info(f"Found {len(test_files)} test result files and {len(cv_files)} CV result files.")

    # --- 2. Concatenate them into master dataframes ---
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
    cv_df = pd.concat([pd.read_csv(f) for f in cv_files], ignore_index=True) if cv_files else pd.DataFrame()

    # --- 3. Save the aggregated dataframes ---
    final_results_dir = os.path.join(results_dir, "aggregated_results")

    test_df.to_csv(os.path.join(final_results_dir, "test_results_all_splits.csv"), index=False)
    logger.info(f"Saved aggregated test results to {os.path.join(final_results_dir, 'test_results_all_splits.csv')}")

    if not cv_df.empty:
        cv_df.to_csv(os.path.join(final_results_dir, "cv_results_all_folds.csv"), index=False)
        logger.info(f"Saved aggregated CV results to {os.path.join(final_results_dir, 'cv_results_all_folds.csv')}")

    # --- 4. Calculate and report summary statistics ---
    summary_df = test_df.drop(columns=['experiment_id']).describe().loc[['mean', 'std']]
    summary_path = os.path.join(final_results_dir, "final_summary_metrics.csv")
    summary_df.to_csv(summary_path)
    
    logger.info("\n--- Aggregated Metrics (Mean ± Std) ---\n" + summary_df.to_string(float_format="%.4f") + "\n" + "---" * 20)
    
    if not cv_df.empty:
        logger.info("--- Generating Validation Metric Distribution Plot ---")
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=cv_df, x='validation_metric', inner='quartile', color='lightblue')
        sns.stripplot(data=cv_df, x='validation_metric', color='darkblue', alpha=0.4, jitter=0.1)
        metric_name = "Metric Value"
        plt.title(f'Distribution of Validation {metric_name} Across All Splits, Folds & Trials', fontsize=16)
        plt.xlabel(f'Validation {metric_name}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        dist_path = os.path.join(final_results_dir, 'validation_metric_distribution.png')
        plt.savefig(dist_path, dpi=300)
        plt.close()
        logger.info(f"Saved validation metric distribution plot to {dist_path}")


def combine_optuna_studies(results_dir: str, output_db_path: str) -> str:
    """
    Finds all Optuna study databases and combines them into a single database.

    Args:
        results_dir (str): The base directory containing experiment subdirectories.
        output_db_path (str): Path to save the new combined SQLite database.

    Returns:
        str: The study name used in the combined database.
    """
    study_files = glob.glob(os.path.join(results_dir, "experiment_*", "optuna_study.db"))
    if not study_files:
        logger.warning("No 'optuna_study.db' files found. Skipping Optuna analysis.")
        return None, None

    logger.info(f"Found {len(study_files)} Optuna study databases to combine.")

    combined_study_name = "aggregated_study"
    storage_url = f"sqlite:///{output_db_path}"
    
    if os.path.exists(output_db_path):
        os.remove(output_db_path)
        logger.info(f"Removed existing aggregated database at {output_db_path}")

    combined_study = optuna.create_study(
        study_name=combined_study_name,
        storage=storage_url,
        direction="maximize" # Or "minimize", depending on your metric
    )

    for db_file in study_files:
        try:
            source_storage = f"sqlite:///{db_file}"
            summaries = optuna.get_all_study_summaries(storage=source_storage)
            
            if not summaries:
                logger.warning(f"No studies found in {db_file}. Skipping.")
                continue
            
            study_name_to_load = summaries[0].study_name

            source_study = optuna.load_study(study_name=study_name_to_load, storage=source_storage)
            combined_study.add_trials(source_study.trials)
            logger.info(f"Successfully added {len(source_study.trials)} trials from {db_file} (Study: '{study_name_to_load}')")
        except Exception as e:
            logger.error(f"Could not process study from {db_file}. Error: {e}")

    logger.info(f"Finished combining all trials into {output_db_path}")
    return combined_study, combined_study_name


# NEW/MODIFIED: The function signature and entire body are updated.
def generate_visualizations(results_dir: str, combined_db_path: str, combined_study_name: str, output_dir: str):
    """
    Generates and saves plots from the combined Optuna study and individual studies.
    - Param Importance and Slice plots use the combined study.
    - Optimization History plot is a subplot of each individual study.
    """
    logger.info("--- Generating Final Analysis Visualizations ---")
    
    if not combined_db_path or not combined_study_name:
        logger.warning("Combined study path or name not provided. Skipping Optuna plots.")
        return

    # --- 1. Generate plots from the COMBINED study ---
    try:
        storage_url = f"sqlite:///{combined_db_path}"
        study_to_plot = optuna.load_study(study_name=combined_study_name, storage=storage_url)
        
        # Plots that benefit from the combined data
        plots_combined = {
            'optuna_param_importances.html': optuna.visualization.plot_param_importances,
            'optuna_slice_plot.html': optuna.visualization.plot_slice,
        }

        for fname, plot_func in plots_combined.items():
            if fname == 'optuna_param_importances.html' and len(study_to_plot.trials) <= 1:
                logger.warning("Skipping parameter importance plot: not enough trials.")
                continue
            fig = plot_func(study_to_plot)
            path = os.path.join(output_dir, fname)
            fig.write_html(path)
            logger.info(f"Saved {fname} to {path}")

    except Exception as e:
        logger.error(f"Could not generate combined Optuna plots. Error: {e}")

    # --- 2. Generate Optimization History as SUBPLOTS from INDIVIDUAL studies ---
    logger.info("Generating subplot optimization history from individual studies...")
    try:
        study_files = glob.glob(os.path.join(results_dir, "experiment_*", "optuna_study.db"))
        if not study_files:
            logger.warning("No individual 'optuna_study.db' files found for history subplots.")
            return

        num_studies = len(study_files)
        cols = int(math.ceil(math.sqrt(num_studies)))
        rows = int(math.ceil(num_studies / cols))
        
        # Get study names for subplot titles
        subplot_titles = []
        for db_file in study_files:
            try:
                s_storage = f"sqlite:///{db_file}"
                s_summary = optuna.get_all_study_summaries(storage=s_storage)
                subplot_titles.append(s_summary[0].study_name if s_summary else os.path.basename(os.path.dirname(db_file)))
            except Exception:
                subplot_titles.append(os.path.basename(os.path.dirname(db_file)))

        fig_main = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        
        current_row, current_col = 1, 1
        for db_file in study_files:
            try:
                source_storage = f"sqlite:///{db_file}"
                study_name = optuna.get_all_study_summaries(storage=source_storage)[0].study_name
                individual_study = optuna.load_study(study_name=study_name, storage=source_storage)
                
                # Generate the single history plot in memory
                fig_individual = optuna.visualization.plot_optimization_history(individual_study)
                
                # Add the traces (the actual lines/dots) from the individual plot to the main subplot figure
                for trace in fig_individual.data:
                    fig_main.add_trace(trace, row=current_row, col=current_col)

                # Move to the next subplot position
                current_col += 1
                if current_col > cols:
                    current_col = 1
                    current_row += 1
            except Exception as e:
                logger.error(f"Could not process study {db_file} for subplot. Error: {e}")
                # Also advance subplot position on error to not overwrite
                current_col += 1
                if current_col > cols:
                    current_col = 1
                    current_row += 1

        fig_main.update_layout(
            title_text='Optimization History per Study',
            height=350 * rows, # Adjust height based on number of rows
            showlegend=False   # Hide individual legends to avoid clutter
        )
        path = os.path.join(output_dir, 'optuna_optimization_history_subplots.html')
        fig_main.write_html(path)
        logger.info(f"Saved subplot optimization history to {path}")

    except Exception as e:
        logger.error(f"Could not generate Optuna history subplots. Error: {e}")


    logger.info("Generating 'Slice Plot of Champions' from each study's best trial...")
    try:
        study_files = glob.glob(os.path.join(results_dir, "experiment_*", "optuna_study.db"))
        if not study_files:
            logger.warning("No individual 'optuna_study.db' files found for best trials plot.")
        else:
            best_trials = []
            for db_file in study_files:
                try:
                    source_storage = f"sqlite:///{db_file}"
                    study_name = optuna.get_all_study_summaries(storage=source_storage)[0].study_name
                    individual_study = optuna.load_study(study_name=study_name, storage=source_storage)
                    best_trials.append(individual_study.best_trial)
                except Exception as e:
                    logger.error(f"Could not load best trial from {db_file}. Error: {e}")

            if best_trials:
                # Create a temporary in-memory study to hold only the best trials for plotting
                best_trials_study = optuna.create_study(
                    direction="maximize", # Make sure this matches your optimization direction
                    study_name="Best Trials Analysis"
                )
                best_trials_study.add_trials(best_trials)

                fig = optuna.visualization.plot_slice(best_trials_study)
                fig.update_layout(title_text="Slice Plot of Best Trial from Each Experiment (Champions Plot)")
                path = os.path.join(output_dir, 'optuna_slice_plot_best_trials.html')
                fig.write_html(path)
                logger.info(f"Saved best trials slice plot to {path}")
            else:
                logger.warning("No best trials were collected, skipping 'Champions Plot'.")

    except Exception as e:
        logger.error(f"Could not generate best trials slice plot. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results from parallel experiments.")
    parser.add_argument('--results_folder_name', type=str, required=True, help="The base directory containing the 'experiment_*' subfolders.")
    args = parser.parse_args()

    results_dir = os.path.join("output/experiments", args.results_folder_name)
    aggregated_results_dir = os.path.join(results_dir, "aggregated_results")
    os.makedirs(aggregated_results_dir, exist_ok=True)
    combined_db_path = os.path.join(aggregated_results_dir, "aggregated_study.db")

    aggregate_and_report(results_dir)

    _, combined_study_name = combine_optuna_studies(
        results_dir=results_dir,
        output_db_path=combined_db_path)
    
    generate_visualizations(
        results_dir=results_dir,
        combined_db_path=combined_db_path,
        combined_study_name=combined_study_name,
        output_dir=aggregated_results_dir)