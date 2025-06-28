import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
import pathlib # Using pathlib is perfect for this task

def plot_missing_values(
    df: pd.DataFrame, 
    df_name: Optional[str] = None,
    save: bool = False,
    output_dir: Optional[Union[str, pathlib.Path]] = None
) -> None:
    """
    Generates and displays or saves a horizontal bar plot showing the 
    percentage of missing values for each column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        df_name (Optional[str]): The name of the DataFrame. Used for the plot
                                 title and for the auto-generated filename
                                 if `save` is True.
        save (bool): If True, the plot is saved to a file. Defaults to False.
        output_dir (Optional[Union[str, pathlib.Path]]): The directory where 
                    the plot will be saved. Required if `save` is True.
    """
    if save:
        if not output_dir:
            raise ValueError("An 'output_dir' must be provided when 'save' is True.")
        if not df_name:
            raise ValueError("A 'df_name' must be provided when 'save' is True to generate the filename.")

    # 1. Calculate missing values
    if df.empty:
        print("The DataFrame is empty. No plot to generate.")
        return

    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / total_rows) * 100
    
    analysis_df = pd.DataFrame({
        'count': missing_counts,
        'percentage': missing_percentage
    }).sort_values(by='percentage', ascending=False)

    # 2. Set up the plot
    num_features = len(analysis_df)
    fig_height = max(8, num_features * 0.3) 

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # 3. Create the horizontal bar plot
    sns.barplot(x='percentage', y=analysis_df.index, data=analysis_df, ax=ax, palette='viridis', hue=analysis_df.index, legend=False)

    # 4. Add annotations
    for bar in ax.patches:
        feature_name = bar.get_y() + bar.get_height() / 2
        count_label = analysis_df.loc[ax.get_yticklabels()[int(round(feature_name))].get_text(), 'count']

        if count_label > 0:
            ax.text(
                x=bar.get_width() + 0.5,
                y=bar.get_y() + bar.get_height() / 2,
                s=f'{int(count_label)}',
                ha='left',
                va='center',
                fontsize=10,
                color='dimgray'
            )
    
    # 5. Set plot title and labels
    plot_title = f"Missing Value Analysis for {df_name}" if df_name else "Missing Value Analysis"
    full_plot_title = f"{plot_title}\n[Total Rows: {total_rows}]"
    
    ax.set_title(full_plot_title, fontsize=16, pad=20)
    ax.set_xlabel("Missing Value Percentage (%)", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    
    # 6. Save the plot with an auto-generated name
    if save:
        # Convert output_dir to a Path object for robust path handling
        output_directory = pathlib.Path(output_dir)
        
        # Create the directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Generate the filename and combine it with the directory path
        filename = f"{df_name}.png"
        full_save_path = output_directory / filename
        
        # Save the figure
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {full_save_path}")
        plt.close(fig)
    else:
        plt.show()