import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import calendar
import numpy as np
from typing import Optional, List, Tuple, Dict
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


class TemporalVisualizerMixin:
    """
    A mixin class for visualizing temporal data from the OfficeGraph.
    
    This class can visualize data from both device_level_df and room_level_df,
    with optional filtering for workhours.
    """
    
    def _get_dataframe_type(self, df: pd.DataFrame) -> str:
        """
        Determine if the dataframe is device-level or room-level based on columns.
        """
        if 'device_uri_str' in df.columns:
            return 'device'
        elif 'room_uri_str' in df.columns:
            return 'room'
        else:
            raise ValueError("DataFrame type could not be determined. Expected 'device_uri_str' or 'room_uri_str' column.")
    
    def _filter_workhours(self, df: pd.DataFrame, workhours_only: bool = False) -> pd.DataFrame:
        """
        Filter dataframe to include only workhours if requested.
        
        Args:
            df: Input dataframe with bucket_idx
            workhours_only: If True, filter to only workhour buckets
        
        Returns:
            Filtered dataframe
        """
        if not workhours_only:
            return df
        
        if not hasattr(self, 'workhour_labels_df'):
            logger.warning("workhour_labels_df not found. Returning unfiltered data.")
            return df
        
        # Get workhour bucket indices
        workhour_buckets = self.workhour_labels_df[
            self.workhour_labels_df['is_workhour'] == 1
        ]['bucket_idx'].values
        
        # Filter the dataframe
        filtered_df = df[df['bucket_idx'].isin(workhour_buckets)]
        
        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} rows (workhours only)")
        return filtered_df
    
    def _get_property_columns(self, df: pd.DataFrame, property_type: str, df_type: str) -> Dict[str, str]:
        """
        Get the column names for a given property type based on dataframe type.
        
        Returns dict mapping stat names to column names.
        """
        columns = {}
        
        if df_type == 'device':
            # Device-level has simple column names
            base_cols = ['mean', 'std', 'max', 'min', 'count', 'has_measurement']
            for col in base_cols:
                if col in df.columns:
                    columns[col] = col
        else:  # room-level
            # Room-level has prefixed column names like "Temperature_mean"
            prefix = f"{property_type}_"
            for col in df.columns:
                if col.startswith(prefix):
                    stat_name = col[len(prefix):]
                    # Map room-level specific names to standard names
                    if stat_name == 'average_intra_device_variation':
                        columns['std'] = col
                    elif stat_name in ['mean', 'max', 'min', 'count', 'has_measurement']:
                        columns[stat_name] = col
                    elif stat_name == 'n_active_devices':
                        columns['n_active'] = col
        
        return columns

    def plot_property_distributions(
            self, 
            df: Optional[pd.DataFrame] = None,
            properties: List[str] = ["Temperature", "Humidity", "CO2Level"],
            stats: List[str] = ["mean", "min", "max", "std", "count"],
            workhours_only: bool = False,
            title_suffix: str = "",
            save_path: Optional[str] = None
    ) -> None:
        """
        Plots the distributions of specified statistics for given property types.
        
        Args:
            df: DataFrame to visualize (device_level_df or room_level_df). 
                If None, attempts to use device_level_df.
            properties: A list of property types to visualize.
            stats: A list of statistics to plot.
            workhours_only: If True, only plot data from workhours.
            title_suffix: Additional text to add to the title.
            save_path: If provided, saves plots to this path with a property-specific suffix,
                       instead of displaying them.
        """
        # Get dataframe if not provided
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        # Determine dataframe type
        df_type = self._get_dataframe_type(df)
        
        # Filter for workhours if requested
        df = self._filter_workhours(df, workhours_only)
        
        # Add workhour indicator to title if filtering
        workhour_text = " (Workhours Only)" if workhours_only else ""
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        
        for prop in properties:
            # Get relevant data for this property
            if df_type == 'device':
                prop_df = df[df['property_type'] == prop]
            else:  # room-level
                prop_df = df
            
            # Get column mappings for this property
            col_map = self._get_property_columns(prop_df, prop, df_type)
            
            # Filter stats to only those available
            available_stats = [s for s in stats if s in col_map]
            
            if not available_stats:
                logger.warning(f"No available statistics for property '{prop}'")
                continue
            
            # Create figure
            fig, axes = plt.subplots(1, len(available_stats), figsize=(5*len(available_stats), 5))
            if len(available_stats) == 1:
                axes = [axes]
            
            title = f"{df_type_text} Distribution of {prop} Statistics{workhour_text}"
            if title_suffix:
                title += f" - {title_suffix}"
            fig.suptitle(title, fontsize=16)
            
            for i, stat in enumerate(available_stats):
                ax = axes[i]
                col_name = col_map[stat]
                
                # Get data and remove NaN values
                data = prop_df[col_name].dropna()
                
                if data.empty:
                    ax.set_title(f"No data for {stat}")
                    ax.axis('off')
                    continue
                
                # Plot histogram
                sns.histplot(data, kde=True, ax=ax)
                ax.set_title(f"{stat.capitalize()} Distribution")
                ax.set_xlabel(f"{prop} {stat}")
                ax.set_ylabel("Frequency")
                
                # Add statistics text
                mean_val = data.mean()
                std_val = data.std()
                ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if save_path:
                # Ensure directory exists
                if os.path.dirname(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Create a unique path for each property
                base, ext = os.path.splitext(save_path)
                prop_save_path = f"{base}_{prop}{ext}"
                
                plt.savefig(prop_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {prop_save_path}")
                plt.close(fig) # Close the figure to free memory
            else:
                plt.show()
    
    def plot_all_distributions_for_room_level_df(
            self,
            df: Optional[pd.DataFrame] = None,
            output_dir: str = "output/distributions",
            id_cols: List[str] = ["bucket_idx", "room_uri_str"],
            workhours_only: bool = False,
            pdf_name: str = "all_distributions.pdf"
    ) -> None:
        """
        Plot distributions for all non-identifier columns in a DataFrame.
        Saves individual PNGs and also a multi-page PDF.

        Args:
            df: DataFrame containing features. If None, uses room_level_df.
            output_dir: Directory to save distribution plots.
            id_cols: Columns to exclude from plotting.
            pdf_name: Name of the combined PDF file.
        """
        if df is None:
            if hasattr(self, 'room_level_df'):
                df = self.room_level_df
            else:
                raise ValueError("No DataFrame provided and room_level_df not available.")

        df = self._filter_workhours(df, workhours_only)

        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, pdf_name)

        with PdfPages(pdf_path) as pdf:
            for col in df.columns:
                if col in id_cols:
                    continue

                plt.figure(figsize=(8, 5))

                # Numeric → histogram + KDE
                if pd.api.types.is_numeric_dtype(df[col]):
                    sns.histplot(df[col].dropna(), kde=True, bins=50, color="steelblue")
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                else:
                    # Categorical → barplot
                    counts = df[col].value_counts()
                    sns.barplot(
                        x=counts.index.astype(str),
                        y=counts.values,
                        color="steelblue"
                    )
                    plt.xticks(rotation=45, ha="right")
                    plt.xlabel(col)
                    plt.ylabel("Count")

                plt.title(f"Distribution of {col}")
                plt.tight_layout()

                # Save PNG
                png_path = os.path.join(output_dir, f"{col}_distribution.png")
                plt.savefig(png_path, dpi=150)

                # Save to PDF
                pdf.savefig()
                plt.close()

        logger.info(f"Saved individual PNGs in '{output_dir}' and a combined PDF at '{pdf_path}'")

    def plot_monthly_daily_trends(
            self, 
            df: Optional[pd.DataFrame] = None,
            properties: List[str] = ["Temperature", "Humidity", "CO2Level"],
            workhours_only: bool = False,
            aggregate_rooms: bool = True,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualizes the average daily trend for each month.
        
        Args:
            df: DataFrame to visualize. If None, uses device_level_df.
            properties: A list of property types to visualize.
            workhours_only: If True, only plot data from workhours.
            aggregate_rooms: For room-level data, whether to aggregate across all rooms.
            save_path: If provided, saves plots to this path with a property-specific suffix,
                       instead of displaying them.
        """
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        if not hasattr(self, 'time_buckets'):
            raise ValueError("time_buckets not available.")
        
        df_type = self._get_dataframe_type(df)
        df = self._filter_workhours(df, workhours_only)
        
        # Add time information
        time_info_df = pd.DataFrame(
            [{'hour_of_day': t[0].hour, 'month': t[0].month} for t in self.time_buckets]
        ).reset_index().rename(columns={'index': 'bucket_idx'})
        
        df = pd.merge(df, time_info_df, on='bucket_idx')
        
        workhour_text = " (Workhours Only)" if workhours_only else ""
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        
        for prop in properties:
            # Get data and column mapping
            if df_type == 'device':
                prop_df = df[df['property_type'] == prop]
                value_col = 'mean'
            else:  # room-level
                col_map = self._get_property_columns(df, prop, df_type)
                if 'mean' not in col_map:
                    logger.warning(f"No mean column found for {prop}")
                    continue
                prop_df = df
                value_col = col_map['mean']
            
            if prop_df.empty or value_col not in prop_df.columns:
                logger.warning(f"No data found for property '{prop}'")
                continue
            
            # Create figure
            fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
            title = f"{df_type_text} Average Daily Trend of {prop} by Month{workhour_text}"
            fig.suptitle(title, fontsize=20)
            
            axes = axes.flatten()
            
            for month in range(1, 13):
                ax = axes[month - 1]
                month_df = prop_df[prop_df['month'] == month]
                
                ax.set_box_aspect(1)
                month_name = calendar.month_name[month]
                ax.set_title(month_name)
                
                if month_df.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # For room-level, optionally aggregate across rooms
                if df_type == 'room' and aggregate_rooms:
                    plot_data = month_df.groupby('hour_of_day')[value_col].agg(['mean', 'std']).reset_index()
                    ax.plot(plot_data['hour_of_day'], plot_data['mean'], marker='o')
                    ax.fill_between(plot_data['hour_of_day'], 
                                   plot_data['mean'] - plot_data['std'],
                                   plot_data['mean'] + plot_data['std'],
                                   alpha=0.3)
                else:
                    sns.lineplot(data=month_df, x='hour_of_day', y=value_col, 
                               ax=ax, errorbar=('ci', 95))
                
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlim(0, 23)
                
                # Add labels to outer plots only
                row = (month - 1) // 4
                col = (month - 1) % 4
                
                if row == 2:  # Bottom row
                    ax.set_xlabel("Hour of Day")
                else:
                    ax.set_xlabel("")
                    
                if col == 0:  # Leftmost column
                    ax.set_ylabel(f"Avg. {prop}")
                else:
                    ax.set_ylabel("")
            
            if save_path:
                if os.path.dirname(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                base, ext = os.path.splitext(save_path)
                prop_save_path = f"{base}_{prop}{ext}"
                plt.savefig(prop_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {prop_save_path}")
                plt.close(fig)
            else:
                plt.show()
    
    def plot_optimal_range_summary(
            self, 
            df: Optional[pd.DataFrame] = None,
            optimal_ranges: Dict[str, Tuple[float, float]] = {
                "Temperature": (21, 24),
                "Humidity": (40, 60),
                "CO2Level": (0, 800)
            },
            workhours_only: bool = False,
            save_path: Optional[str] = None
    ) -> None:
        """
        Plots the percentage of time each property is within optimal range.
        
        Args:
            df: DataFrame to analyze. If None, uses device_level_df.
            optimal_ranges: Dictionary of property names to (min, max) tuples.
            workhours_only: If True, only analyze workhours.
            save_path: If provided, saves the plot to this path instead of displaying it.
        """
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        df_type = self._get_dataframe_type(df)
        df = self._filter_workhours(df, workhours_only)
        
        summary_data = []
        
        for prop, (min_val, max_val) in optimal_ranges.items():
            # Get data and column mapping
            if df_type == 'device':
                prop_df = df[df['property_type'] == prop].dropna(subset=['mean'])
                if prop_df.empty:
                    continue
                values = prop_df['mean']
            else:  # room-level
                col_map = self._get_property_columns(df, prop, df_type)
                if 'mean' not in col_map:
                    continue
                values = df[col_map['mean']].dropna()
                if values.empty:
                    continue
            
            # Calculate percentages
            in_range = ((values >= min_val) & (values <= max_val)).sum()
            total = len(values)
            pct_in = (in_range / total) * 100
            pct_out = 100 - pct_in
            
            summary_data.append({'property': prop, 'status': 'In Range', 'percentage': pct_in})
            summary_data.append({'property': prop, 'status': 'Out of Range', 'percentage': pct_out})
        
        if not summary_data:
            logger.error("No data to plot")
            return
        
        # Create plot
        results_df = pd.DataFrame(summary_data)
        pivot_df = results_df.pivot(index='property', columns='status', values='percentage')
        pivot_df = pivot_df[['In Range', 'Out of Range']]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {'In Range': '#2ca02c', 'Out of Range': '#d62728'}
        
        pivot_df.plot(kind='bar', stacked=True, ax=ax,
                     color=[colors.get(c, '#333333') for c in pivot_df.columns])
        
        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fmt='%.1f%%',
                        color='white', weight='bold')
        
        # Title and labels
        workhour_text = " (Workhours Only)" if workhours_only else ""
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        ax.set_title(f'{df_type_text} Compliance with Optimal Environmental Ranges{workhour_text}',
                    fontsize=16, pad=20)
        ax.set_ylabel('Percentage of Time (%)', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=0, ha='center', fontsize=12)
        ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_heatmap(
            self,
            df: Optional[pd.DataFrame] = None,
            properties: List[str] = ["Temperature", "Humidity", "CO2Level"],
            stats_to_correlate: List[str] = ["mean", "std"],
            workhours_only: bool = False,
            save_path: Optional[str] = None
    ) -> None:
        """
        Plots a correlation heatmap between sensor properties.
        
        Args:
            df: DataFrame to analyze. If None, uses device_level_df.
            properties: Properties to include in correlation.
            stats_to_correlate: Statistics to correlate.
            workhours_only: If True, only analyze workhours.
            save_path: If provided, saves the plot to this path instead of displaying it.
        """
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        df_type = self._get_dataframe_type(df)
        df = self._filter_workhours(df, workhours_only)
        
        if df_type == 'device':
            # Filter and aggregate device-level data
            df_filtered = df[df['property_type'].isin(properties)][
                ['bucket_idx', 'property_type'] + stats_to_correlate
            ]
            df_aggregated = df_filtered.groupby(['bucket_idx', 'property_type']).mean().reset_index()
            
            # Pivot to wide format
            df_wide = df_aggregated.pivot(
                index='bucket_idx',
                columns='property_type',
                values=stats_to_correlate
            )
            df_wide.columns = [f'{col[1]}_{col[0]}' for col in df_wide.columns]
            
        else:  # room-level
            # For room-level, collect relevant columns
            columns_to_use = ['bucket_idx']
            column_labels = []
            
            for prop in properties:
                col_map = self._get_property_columns(df, prop, df_type)
                for stat in stats_to_correlate:
                    if stat in col_map:
                        columns_to_use.append(col_map[stat])
                        # Use simpler labels for room-level
                        if stat == 'std':
                            column_labels.append(f'{prop}_intra_device_var')
                        else:
                            column_labels.append(f'{prop}_{stat}')
            
            # Aggregate across rooms for each bucket
            df_wide = df[columns_to_use].groupby('bucket_idx').mean()
            df_wide.columns = column_labels
        
        # Calculate correlation
        corr_matrix = df_wide.corr()
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        workhour_text = " (Workhours Only)" if workhours_only else ""
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='vlag',
                   vmin=-1, vmax=1, square=True)
        
        plt.title(f'{df_type_text} Correlation Matrix of Sensor Properties{workhour_text}',
                 fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_missing_data_pattern(
            self,
            df: Optional[pd.DataFrame] = None,
            properties: List[str] = ["Temperature", "Humidity", "CO2Level"],
            workhours_only: bool = False,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize missing data patterns over time for each property.
        
        Args:
            df: DataFrame to analyze. If None, uses device_level_df.
            properties: Properties to analyze.
            workhours_only: If True, only analyze workhours.
            save_path: If provided, saves the plot to this path instead of displaying it.
        """
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        df_type = self._get_dataframe_type(df)
        df = self._filter_workhours(df, workhours_only)
        
        # Get time information for x-axis
        if hasattr(self, 'time_buckets'):
            time_map = {i: t[0] for i, t in enumerate(self.time_buckets)}
        else:
            time_map = {i: i for i in df['bucket_idx'].unique()}
        
        fig, axes = plt.subplots(len(properties), 1, figsize=(15, 3*len(properties)),
                                sharex=True)
        if len(properties) == 1:
            axes = [axes]
        
        workhour_text = " (Workhours Only)" if workhours_only else ""
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        fig.suptitle(f'{df_type_text} Missing Data Patterns{workhour_text}', fontsize=16)
        
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            
            if df_type == 'device':
                # For device-level, aggregate missing percentage per bucket
                prop_df = df[df['property_type'] == prop]
                missing_by_bucket = prop_df.groupby('bucket_idx')['has_measurement'].apply(
                    lambda x: 100 * (1 - x.mean())
                )
            else:  # room-level
                col_map = self._get_property_columns(df, prop, df_type)
                if 'has_measurement' in col_map:
                    missing_by_bucket = df.groupby('bucket_idx')[col_map['has_measurement']].apply(
                        lambda x: 100 * (1 - x.mean())
                    )
                else:
                    continue
            
            # Convert to time series for plotting
            times = [time_map.get(i, i) for i in missing_by_bucket.index]
            
            ax.fill_between(times, 0, missing_by_bucket.values, alpha=0.3, label=prop)
            ax.plot(times, missing_by_bucket.values, linewidth=1)
            ax.set_ylabel(f'{prop}\nMissing %')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def compare_workhours_distributions(
            self,
            df: Optional[pd.DataFrame] = None,
            properties: List[str] = ["Temperature", "CO2Level"],
            stat: str = "mean",
            save_path: Optional[str] = None
    ) -> None:
        """
        Compare distributions between workhours and non-workhours.
        
        Args:
            df: DataFrame to analyze. If None, uses device_level_df.
            properties: Properties to compare.
            stat: Statistic to compare (e.g., 'mean', 'std').
            save_path: If provided, saves the plot to this path instead of displaying it.
        """
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        if not hasattr(self, 'workhour_labels_df'):
            logger.error("workhour_labels_df not available for comparison")
            return
        
        df_type = self._get_dataframe_type(df)
        
        # Add workhour labels to dataframe
        df = pd.merge(df, self.workhour_labels_df, on='bucket_idx', how='left')
        
        fig, axes = plt.subplots(1, len(properties), figsize=(6*len(properties), 5))
        if len(properties) == 1:
            axes = [axes]
        
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        fig.suptitle(f'{df_type_text} Workhours vs Non-Workhours Comparison ({stat})', fontsize=16)
        
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            
            # Get the appropriate column
            if df_type == 'device':
                prop_df = df[df['property_type'] == prop]
                if stat not in prop_df.columns:
                    continue
                value_col = stat
            else:  # room-level
                col_map = self._get_property_columns(df, prop, df_type)
                if stat not in col_map:
                    continue
                prop_df = df
                value_col = col_map[stat]
            
            # Separate workhours and non-workhours
            workhour_data = prop_df[prop_df['is_workhour'] == 1][value_col].dropna()
            non_workhour_data = prop_df[prop_df['is_workhour'] == 0][value_col].dropna()
            
            # Create violin plot
            plot_data = pd.DataFrame({
                'Value': pd.concat([workhour_data, non_workhour_data]),
                'Type': ['Workhours']*len(workhour_data) + ['Non-Workhours']*len(non_workhour_data)
            })
            
            sns.violinplot(data=plot_data, x='Type', y='Value', ax=ax)
            ax.set_title(f'{prop}')
            ax.set_ylabel(f'{stat.capitalize()} Value')
            ax.set_xlabel('')
            
            # Add mean lines
            for i, (data, label) in enumerate([(workhour_data, 'Workhours'), 
                                               (non_workhour_data, 'Non-Workhours')]):
                if len(data) > 0:
                    mean_val = data.mean()
                    ax.hlines(mean_val, i-0.4, i+0.4, colors='red', linestyles='--', label=f'Mean: {mean_val:.2f}')
            
            # Add statistics text
            if len(workhour_data) > 0 and len(non_workhour_data) > 0:
                from scipy import stats as scipy_stats
                # Perform Welch's t-test (more robust to unequal variances)
                t_stat, p_val = scipy_stats.ttest_ind(workhour_data, non_workhour_data, equal_var=False)
                
                # Format p-value according to scientific convention
                if p_val < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_val:.3f}"
                
                # Combine t-statistic (rounded to 2 decimal places) and p-value
                stats_text = f"t = {t_stat:.2f}\n{p_text}"
                
                ax.text(0.5, 0.95, stats_text, 
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                
        plt.tight_layout()
        
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_full_correlation_heatmap(
            self,
            df: Optional[pd.DataFrame] = None,
            workhours_only: bool = False,
            method: str = 'pearson',
            min_correlation: float = 0.0,
            figsize: Optional[Tuple[int, int]] = None,
            exclude_columns: Optional[List[str]] = None,
            include_only_numeric: bool = True,
            aggregate_by_bucket: bool = True,
            show_top_n: Optional[int] = None,
            annot: bool = True,
            save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Plot a comprehensive correlation heatmap for all numeric columns in the dataframe.
        
        This function is designed to handle both device_level_df and room_level_df,
        showing correlations between ALL features including engineered features,
        static features, time features, etc.
        
        Args:
            df: DataFrame to analyze. If None, uses device_level_df.
            workhours_only: If True, only analyze workhours.
            method: Correlation method ('pearson', 'spearman', or 'kendall').
            min_correlation: Only show correlations with absolute value above this threshold.
            figsize: Figure size. If None, automatically determined based on number of features.
            exclude_columns: List of column names to exclude from correlation analysis.
            include_only_numeric: If True, only include numeric columns.
            aggregate_by_bucket: For device-level df, whether to aggregate by bucket first.
            show_top_n: If specified, only show the top N most correlated feature pairs.
            annot: Whether to annotate the heatmap with correlation values.
            save_path: If provided, saves the plot to this path instead of displaying it.
            
        Returns:
            The correlation matrix as a DataFrame.
        """
        if df is None:
            if hasattr(self, 'device_level_df'):
                df = self.device_level_df
            else:
                raise ValueError("No DataFrame provided and device_level_df not available.")
        
        df_type = self._get_dataframe_type(df)
        df = self._filter_workhours(df, workhours_only)
        
        # Prepare data based on dataframe type
        if df_type == 'device' and aggregate_by_bucket:
            # For device-level, aggregate by bucket to reduce noise
            # Group by bucket_idx and property_type, then pivot
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'bucket_idx' in numeric_cols:
                numeric_cols.remove('bucket_idx')
            
            # Check if property_type exists
            if 'property_type' in df.columns:
                # Pivot to get one row per bucket with columns for each property-stat combination
                agg_dict = {col: 'mean' for col in numeric_cols}
                df_agg = df.groupby(['bucket_idx', 'property_type']).agg(agg_dict).reset_index()
                
                # Create wide format with property_type as column prefix
                df_wide = df_agg.pivot_table(
                    index='bucket_idx',
                    columns='property_type',
                    values=numeric_cols,
                    aggfunc='mean'
                )
                
                # Flatten column names
                df_wide.columns = [f'{prop}_{stat}' if prop else stat 
                                  for stat, prop in df_wide.columns]
                df_to_correlate = df_wide
            else:
                # If no property_type, just aggregate by bucket
                df_to_correlate = df.groupby('bucket_idx').mean()
        else:
            # For room-level or non-aggregated device-level
            df_to_correlate = df.copy()
        
        # Select only numeric columns
        if include_only_numeric:
            numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_cols = df_to_correlate.select_dtypes(include=numeric_dtypes).columns.tolist()
            
            # Remove excluded columns
            if exclude_columns:
                numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
            
            # Also remove index-like columns that shouldn't be correlated
            for col in ['bucket_idx', 'room_uri_str', 'device_uri_str']:
                if col in numeric_cols:
                    numeric_cols.remove(col)
            
            df_to_correlate = df_to_correlate[numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = df_to_correlate.corr(method=method)
        
        # Apply minimum correlation threshold if specified
        if min_correlation > 0:
            # Create a mask for correlations below threshold
            mask = np.abs(corr_matrix) < min_correlation
            corr_matrix_filtered = corr_matrix.copy()
            corr_matrix_filtered[mask] = np.nan
        else:
            corr_matrix_filtered = corr_matrix
        
        # If show_top_n is specified, find and display only top correlations
        if show_top_n:
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_tri = np.triu(corr_matrix.values, k=1)
            upper_tri[upper_tri == 0] = np.nan
            
            # Find top N correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if not np.isnan(upper_tri[i, j]):
                        correlations.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': upper_tri[i, j],
                            'Abs Correlation': abs(upper_tri[i, j])
                        })
            
            correlations_df = pd.DataFrame(correlations)
            if not correlations_df.empty:
                top_correlations = correlations_df.nlargest(show_top_n, 'Abs Correlation')
                
                # Get unique features from top correlations
                top_features = list(set(
                    top_correlations['Feature 1'].tolist() + 
                    top_correlations['Feature 2'].tolist()
                ))
                
                # Filter correlation matrix to only these features
                corr_matrix_filtered = corr_matrix.loc[top_features, top_features]
        
        # Determine figure size if not specified
        if figsize is None:
            n_features = len(corr_matrix_filtered.columns)
            # Scale figure size based on number of features
            size = min(max(10, n_features * 0.5), 30)  # Cap at 30 to avoid too large figures
            figsize = (size, size * 0.8)
        
        # Create the heatmap
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle to avoid redundancy (optional)
        mask_upper = np.triu(np.ones_like(corr_matrix_filtered, dtype=bool), k=1)
        
        # Determine annotation format based on matrix size
        if annot and len(corr_matrix_filtered.columns) > 30:
            annot = False  # Too many features to annotate clearly
            logger.info("Disabling annotations due to large number of features (>30)")
        
        fmt = '.2f' if annot else ''
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix_filtered,
            mask=mask_upper if len(corr_matrix_filtered.columns) < 20 else None,  # Only mask for smaller matrices
            annot=annot,
            fmt=fmt,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5 if len(corr_matrix_filtered.columns) < 30 else 0,
            cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"}
        )
        
        # Set title
        workhour_text = " (Workhours Only)" if workhours_only else ""
        df_type_text = "Device-Level" if df_type == 'device' else "Room-Level"
        title = f'{df_type_text} Feature Correlation Heatmap{workhour_text}'
        if min_correlation > 0:
            title += f' (|r| > {min_correlation})'
        if show_top_n:
            title += f' (Top {show_top_n} Correlations)'
        
        plt.title(title, fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        
        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
        
        # Print summary statistics
        logger.info(f"Correlation matrix shape: {corr_matrix_filtered.shape}")
        logger.info(f"Number of features analyzed: {len(corr_matrix_filtered.columns)}")
        
        # Find and report highest correlations (excluding diagonal)
        if not show_top_n:  # Only if we haven't already shown top N
            upper_tri = np.triu(corr_matrix.values, k=1)
            high_corr_indices = np.where(np.abs(upper_tri) > 0.7)
            if len(high_corr_indices[0]) > 0:
                logger.info("\nHigh correlations (|r| > 0.7):")
                for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    logger.info(f"  {feature1} <-> {feature2}: {corr_value:.3f}")
        
        return corr_matrix
