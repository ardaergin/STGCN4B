import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

logger = logging.getLogger(__name__)

class TabularBuilderMixin:
    """
    Mixin to convert temporal and weather features into a tabular dataset
    for gradient-boosted tree models (LightGBM, XGBoost).
    Assumes TemporalBuilderMixin has already generated:
      - normalized_measurements_df
      - weather_features_norm_
      - train_indices, val_indices, test_indices
      - time_buckets
    """

    def _extract_datetime_features(self) -> pd.DataFrame:
        """
        Build a DataFrame of datetime-derived features for each bucket.
        Returns a DataFrame indexed by bucket_idx with columns: hour, dayofweek, is_weekend
        """
        times = [start for start, _ in self.time_buckets]
        df = pd.DataFrame({
            'bucket_idx': range(len(times)),
            'datetime': times
        })
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(float)
        return df.set_index('bucket_idx')

    def _add_cyclical_features(self, wide: pd.DataFrame) -> pd.DataFrame:
        """
        Add sine/cosine transforms of hour and dayofweek for cyclical encoding.
        """
        dt = self._extract_datetime_features()
        wide = wide.join(dt)
        wide['hour_sin'] = np.sin(2 * np.pi * wide['hour'] / 24)
        wide['hour_cos'] = np.cos(2 * np.pi * wide['hour'] / 24)
        wide['dow_sin']  = np.sin(2 * np.pi * wide['dayofweek'] / 7)
        wide['dow_cos']  = np.cos(2 * np.pi * wide['dayofweek'] / 7)
        return wide.drop(columns=['hour','dayofweek','is_weekend'], errors='ignore')

    def _add_sensor_aggregates(self, wide: pd.DataFrame) -> pd.DataFrame:
        """
        Compute high-level aggregates across sensors: avg temp, max co2, avg humidity.
        """
        temp_cols = [c for c in wide.columns if '__Temperature__mean' in c]
        if temp_cols:
            wide['avg_temperature'] = wide[temp_cols].mean(axis=1)
        co2_cols = [c for c in wide.columns if '__CO2Level__max' in c]
        if co2_cols:
            wide['max_co2'] = wide[co2_cols].max(axis=1)
        hum_cols = [c for c in wide.columns if '__Humidity__mean' in c]
        if hum_cols:
            wide['avg_humidity'] = wide[hum_cols].mean(axis=1)
        return wide

    def _add_lag_features(self, wide: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lag features for specified columns.
        Args:
            wide: DataFrame indexed by bucket_idx.
            cols: List of column names to lag.
            lags: List of integer lag steps (e.g., [1,2,3]).
        Note:
            Missing values are left as NaN so that tree-based models can
            handle them natively; downstream pipelines can choose to
            fill or impute if needed.
        """
        for col in cols:
            for lag in lags:
                wide[f'{col}_lag{lag}'] = wide[col].shift(lag)
        return wide

    def _add_rolling_features(self, wide: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Add rolling-window summary features for specified columns.
        Args:
            wide: DataFrame indexed by bucket_idx.
            cols: List of column names to roll over.
            windows: List of window sizes (in buckets).
        """
        for col in cols:
            for win in windows:
                wide[f'{col}_roll_mean_{win}'] = wide[col].rolling(win, min_periods=1).mean()
                wide[f'{col}_roll_std_{win}']  = wide[col].rolling(win, min_periods=1).std().fillna(0)
        return wide

    def to_tabular(self,
                   drop_sum: bool = True,
                   include_datetime: bool = True,
                   include_sensor_aggs: bool = True,
                   lag_cols: List[str] = None,
                   lag_steps: List[int] = [1,2,3],
                   roll_cols: List[str] = None,
                   roll_windows: List[int] = [3,6,12]
                  ) -> Dict[str, Any]:
        """
        Convert the full graph + weather features into a tabular dataset.

        Args:
            drop_sum: Drop 'sum' feature if True.
            include_datetime: Whether to add cyclical time features.
            include_sensor_aggs: Whether to add sensor aggregate features.
            lag_cols: Columns to generate lag features for. Defaults to ['consumption'].
            lag_steps: Lag steps.
            roll_cols: Columns for rolling summaries. Defaults to sensor aggregates.
            roll_windows: Window sizes.
        """
        full_df = self.build_full_feature_df(drop_sum=drop_sum)
        weather = pd.DataFrame.from_dict(self.weather_features_norm_, orient='index')
        weather.index.name = 'bucket_idx'; weather = weather.reset_index()
        merged = full_df.merge(weather, on='bucket_idx', how='left')  # preserve NaNs for missing measurements and weather

        meas_cols = ['mean','std','max','min','count']
        if not drop_sum: meas_cols.append('sum')
        idc = ['bucket_idx','device_uri','property_type']
        meas_wide = merged[idc+meas_cols].pivot_table(
            index='bucket_idx', columns=['device_uri','property_type'], values=meas_cols  # let missing stay NaN
        )
        meas_wide.columns = [f"{d}__{p}__{f}" for d,p,f in meas_wide.columns]
        wide = meas_wide.reset_index().set_index('bucket_idx')

        # feature engineering
        if include_datetime:
            wide = self._add_cyclical_features(wide)
        if include_sensor_aggs:
            wide = self._add_sensor_aggregates(wide)

        # default lag on consumption target
        if lag_cols is None:
            lag_cols = ['consumption']
        # inject consumption series
        cons = pd.Series(
            [self.get_forecasting_values(True)[i] for i in wide.index],
            index=wide.index,
            name='consumption'
        )
        wide = wide.join(cons)
        if lag_cols:
            wide = self._add_lag_features(wide, lag_cols, lag_steps)

        # rolling on aggregates
        if roll_cols is None:
            roll_cols = [c for c in wide.columns if c.startswith('avg_') or c.startswith('max_')]
        if roll_cols:
            wide = self._add_rolling_features(wide, roll_cols, roll_windows)

        # finalize
        X = wide.values
        feature_names = wide.columns.tolist()
        y = {
            'workhour': self.get_classification_labels(),
            'consumption': cons.values
        }
        return {
            'X': X, 'y': y, 'feature_names': feature_names,
            'train_idx': self.train_indices, 'val_idx': self.val_indices, 'test_idx': self.test_indices
        }
