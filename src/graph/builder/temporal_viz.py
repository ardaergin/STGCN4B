import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import calendar

logger = logging.getLogger(__name__)

class TemporalVisualizerMixin:
    """
    A mixin class for visualizing temporal data from the OfficeGraph.
    
    This class assumes that the data is available in a pandas DataFrame
    called `bucketed_measurements_df` on the instance it's mixed into.
    """

    def plot_property_distributions(self, 
                                    properties: list[str] = ["Temperature", "Humidity", "CO2Level"],
                                    stats: list[str] = ["mean", "min", "max", "std"]) -> None:
        """
        Plots the distributions of specified statistics for given property types.

        This creates a grid of histograms to show how the mean, min, and max
        values are distributed for properties like Temperature, Humidity, and CO2.
        It only considers the data points that have actual measurements.

        Args:
            properties: A list of property types to visualize (e.g., ["Temperature", "CO2"]).
            stats: A list of statistics to plot (e.g., ["mean", "max"]).
        """
        if not hasattr(self, 'bucketed_measurements_df'):
            raise ValueError("The 'bucketed_measurements_df' DataFrame is not available.")
        
        df = self.bucketed_measurements_df

        for prop in properties:
            # Create a figure with subplots for each statistic
            fig, axes = plt.subplots(1, len(stats), figsize=(15, 5))
            fig.suptitle(f"Distribution of {prop} Statistics", fontsize=16)

            for i, stat in enumerate(stats):
                ax = axes[i]
                data = df[df['property_type'] == prop][stat]
                
                if data.empty:
                    logger.warning(f"No data for property '{prop}' and stat '{stat}'.")
                    ax.set_title(f"No data for {stat}")
                    continue

                # Use seaborn for a nice-looking histogram
                sns.histplot(data, kde=True, ax=ax)
                ax.set_title(f"{stat.capitalize()} Distribution")
                ax.set_xlabel(f"{prop} {stat}")
                ax.set_ylabel("Frequency")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    def plot_monthly_daily_trends(self, 
                                  properties: list[str] = ["Temperature", "Humidity", "CO2Level"]) -> None:
        """
        Visualizes the average daily trend for each month in a separate, square subplot.

        For each property, this function creates a 3x4 grid of plots. Each subplot
        is forced into a square aspect ratio and shows the average hourly trend for
        a specific month. Axis labels are only shown on the outer plots to avoid clutter.

        Args:
            properties: A list of property types to visualize.
        """
        if not hasattr(self, 'bucketed_measurements_df'):
            raise ValueError("The 'bucketed_measurements_df' DataFrame is not available.")
        if not hasattr(self, 'time_buckets'):
            raise ValueError("The 'time_buckets' are not available.")
            
        # 1. Prepare the DataFrame by adding month and hour information
        time_info_df = pd.DataFrame(
            [{'hour_of_day': t[0].hour, 'month': t[0].month} for t in self.time_buckets]
        ).reset_index().rename(columns={'index': 'bucket_idx'})
        
        df = pd.merge(self.bucketed_measurements_df, time_info_df, on='bucket_idx')

        # 2. Loop through each property to create a separate figure
        for prop in properties:
            prop_df = df[df['property_type'] == prop]

            if prop_df.empty:
                logger.warning(f"No data found for property '{prop}'. Skipping plot.")
                continue

            # Create a 3x4 grid. 
            # `constrained_layout=True` automatically adjusts spacing.
            # `figsize` is adjusted to better fit square plots.
            fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
            fig.suptitle(f"Average Daily Trend of {prop} by Month", fontsize=20)
            
            axes = axes.flatten()

            # 3. Loop through each month and draw a subplot
            for month in range(1, 13):
                ax = axes[month - 1]
                month_df = prop_df[prop_df['month'] == month]
                
                # --- Improvement 1: Enforce a square aspect ratio ---
                ax.set_box_aspect(1)
                
                month_name = calendar.month_name[month]
                ax.set_title(month_name)

                if month_df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    continue

                sns.lineplot(
                    data=month_df, 
                    x='hour_of_day', 
                    y='mean', 
                    ax=ax,
                    errorbar=('ci', 95)
                )
                
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                
                # --- Improvement 2: Add labels to outer plots only ---
                row = (month - 1) // 4
                col = (month - 1) % 4
                
                if row == 2:  # Bottom row
                    ax.set_xlabel("Hour of Day")
                if col == 0:  # Leftmost column
                    ax.set_ylabel(f"Avg. {prop}")

            plt.show()

    def plot_optimal_range_summary(self, 
                                 optimal_ranges: dict = {
                                     "Temperature": (21, 24),  # Celsius
                                     "Humidity": (40, 60),     # Percent
                                     "CO2Level": (0, 800)     # PPM
                                 }) -> None:
        """
        Calculates and plots the percentage of time each property is within a defined optimal range.

        This function generates a stacked bar chart (a "poll-like" plot) showing the
        proportion of measurements that fall inside vs. outside the specified optimal
        range for each property.

        Args:
            optimal_ranges: A dictionary where keys are property names and values are
                            a tuple (min_optimal, max_optimal).
        """
        if not hasattr(self, 'bucketed_measurements_df'):
            raise ValueError("The 'bucketed_measurements_df' DataFrame is not available.")

        # 1. Calculate the percentages for each property
        summary_data = []
        for prop, (min_val, max_val) in optimal_ranges.items():
            # Filter for the specific property and drop any rows without a 'mean' value
            prop_df = self.bucketed_measurements_df[
                self.bucketed_measurements_df['property_type'] == prop
            ].dropna(subset=['mean'])

            if prop_df.empty:
                logger.warning(f"No data found for property '{prop}'. Skipping.")
                continue

            # Check which measurements are within the optimal range
            in_range_mask = (prop_df['mean'] >= min_val) & (prop_df['mean'] <= max_val)
            
            total_count = len(prop_df)
            in_count = in_range_mask.sum()

            # Calculate percentages
            pct_in = (in_count / total_count) * 100
            pct_out = 100 - pct_in

            summary_data.append({'property': prop, 'status': 'In Range', 'percentage': pct_in})
            summary_data.append({'property': prop, 'status': 'Out of Range', 'percentage': pct_out})

        if not summary_data:
            logger.error("Could not compute summary. No data matched the specified ranges.")
            return

        # 2. Prepare the data for plotting by creating a DataFrame and pivoting it
        results_df = pd.DataFrame(summary_data)
        
        # Pivot the table to get properties as rows and status as columns
        pivot_df = results_df.pivot(index='property', columns='status', values='percentage')
        
        # Ensure the column order is consistent for stacking
        pivot_df = pivot_df[['In Range', 'Out of Range']]

        # 3. Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Using specific, intuitive colors for "In" and "Out" of range
        colors = {'In Range': '#2ca02c', 'Out of Range': '#d62728'} # Green and Red
        
        pivot_df.plot(
            kind='bar', 
            stacked=True, 
            ax=ax,
            color=[colors.get(c, '#333333') for c in pivot_df.columns] # Use our defined colors
        )
        
        # 4. Add percentage labels onto the bars for clarity
        for container in ax.containers:
            ax.bar_label(
                container, 
                label_type='center', 
                fmt='%.1f%%', # Format as a percentage with one decimal place
                color='white',
                weight='bold'
            )
        
        # 5. Finalize plot aesthetics
        ax.set_title('Compliance with Optimal Environmental Ranges', fontsize=16, pad=20)
        ax.set_ylabel('Percentage of Time (%)', fontsize=12)
        ax.set_xlabel('') # Property name is clear from the x-ticks
        ax.set_ylim(0, 100)
        plt.xticks(rotation=0, ha='center', fontsize=12) # Keep labels horizontal
        
        # Move legend outside the plot
        ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.show()

    def plot_correlation_heatmap(self,
                                properties: list[str] = ["Temperature", "Humidity", "CO2Level"],
                                stats_to_correlate: list[str] = ["mean", "std"]) -> None:
        """
        Plots a heatmap of the correlations between sensor properties.

        This function reshapes the data to create a wide-format DataFrame where each
        row is a time bucket and columns represent a specific stat (e.g., 'Temperature_mean',
        'Humidity_std'). It then computes and visualizes the Pearson correlation
        coefficient between all pairs of these columns.

        Args:
            properties: A list of property types to include in the correlation matrix.
            stats_to_correlate: The statistics to use for correlation ('mean', 'std', etc.).
        """
        if not hasattr(self, 'bucketed_measurements_df'):
            raise ValueError("The 'bucketed_measurements_df' DataFrame is not available.")

        # 1. Reshape the data from a long to a wide format.
        df_long = self.bucketed_measurements_df
        
        df_filtered = df_long[
            df_long['property_type'].isin(properties)
        ][['bucket_idx', 'property_type'] + stats_to_correlate]

        # --- THE FIX IS HERE ---
        # Before pivoting, must aggregate across devices for each property within a bucket.
        # Can take the mean of the values (e.g., the average of all 'mean' temperature readings).
        df_aggregated = df_filtered.groupby(['bucket_idx', 'property_type']).mean().reset_index()
        # -----------------------

        # Pivot the aggregated table. This will now work without duplicates.
        df_wide = df_aggregated.pivot(
            index='bucket_idx',
            columns='property_type',
            values=stats_to_correlate
        )
        
        # The pivot creates a multi-level column index. So, flatten it.
        df_wide.columns = [f'{col[1]}_{col[0]}' for col in df_wide.columns]

        # 2. Calculate the correlation matrix
        corr_matrix = df_wide.corr()

        # 3. Plot the heatmap
        plt.figure(figsize=(10, 8))
        
        heatmap = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='vlag',
            vmin=-1, vmax=1
        )
        
        heatmap.set_title('Correlation Matrix of Sensor Properties', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()