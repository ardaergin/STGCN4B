from typing import List

from ...preparation.feature import BlockAwareFeatureEngineer

import logging; logger = logging.getLogger(__name__)


class TabularBuilderMixin:
        
    def build_base_tabular_df(self, build_mode: str) -> None:
        """
        Selects and prepares the initial base DataFrame based on the `build_mode`.
        This method populates `self.tabular_feature_df` with room, floor, or building level data.
        """
        logger.info(f"Building base tabular DataFrame for build_mode: '{build_mode}'")
        
        if build_mode == "consumption_forecast":
            # NOTE: For this task, we have a single value per time bucket for the consumption of the whole building. 
            #       Using all devices (160~), and adding MA and lag, etc. would be too much features
            #       We can use either the floor-level or building-level DataFrames.
            #       floor-level DataFrame is still a lot of features. So, using building-level.
            base_df = self.building_level_df.copy()
        
        elif build_mode == "measurement_forecast":            
            base_df = self.add_static_room_features_to_df(df=self.room_level_df)

        else:
            raise ValueError(f"Unknown build_mode: {build_mode}")
        
        self.tabular_feature_df = base_df
        logger.info(f"Base DataFrame built. Shape: {self.tabular_feature_df.shape}")
        return None
    
    def engineer_tabular_features(self,
                                build_mode: str,
                                lags: List[int] = None,
                                windows: List[int] = None,
                                shift_amount: int = 1, 
                                integrate_weather: bool = True):
        """
        Applies feature engineering (lags, MAs, time, weather) to the existing `tabular_feature_df`.
        """
        if not hasattr(self, 'tabular_feature_df'):
            raise ValueError("tabular_feature_df not found. Run build_base_tabular_df() first.")

        # Instantiating the feature engineer
        feature_engineer = BlockAwareFeatureEngineer(self.blocks)

        # Adding weather features here so we get also their MAs and lags
        if integrate_weather:
            self.tabular_feature_df = self.add_weather_features_to_df(df=self.tabular_feature_df)
        
        # Defining selective feature lists for lags and moving averages
        base_cols = self.tabular_feature_df.select_dtypes("number").columns.tolist()
        base_cols = [c for c in base_cols if c not in ('bucket_idx', 'block_id')]

        # Tier 1: Core signals for both Lags and MAs
        core_signals_for_lags_and_ma = [
            c for c in base_cols if 
            any(k in c for k in ['_mean', '_max', '_min']) or
            c in ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                  'wind_speed_10m', 'wind_speed_80m', 'cloud_cover']
        ]
        logger.info(f"Generating lags and MAs for {len(core_signals_for_lags_and_ma)} core signal columns.")

        # Tier 2: Secondary signals for MA only
        secondary_signals_for_ma_only = [
            c for c in base_cols if 
            any(k in c for k in ['_std', '_count', '_n_devices', '_has_measurement'])
        ]
        logger.info(f"Generating MAs only for {len(secondary_signals_for_ma_only)} secondary signal columns.")

        # Additional grouping col for "measurement_forecast" task
        extra_grouping_col = ["room_uri_str"] if build_mode == "measurement_forecast" else None

        # Defaults for MAs & Lags
        lags = lags or [1, 2, 3, 24]
        windows = windows or [3, 6, 12, 24]

        # Creating MAs & Lags        
        self.tabular_feature_df = feature_engineer.add_lag_features(
            lags=lags,
            data_frame=self.tabular_feature_df,
            cols=core_signals_for_lags_and_ma,
            use_only_original_columns=True,
            extra_grouping_cols=extra_grouping_col)
        
        self.tabular_feature_df = feature_engineer.add_moving_average_features(
            windows=windows,
            shift_amount=shift_amount,
            data_frame=self.tabular_feature_df,
            cols=core_signals_for_lags_and_ma + secondary_signals_for_ma_only,
            use_only_original_columns=True,
            extra_grouping_cols=extra_grouping_col)

        # NOTE: We can add the time features after taking MA & lag,
        #       as we should not really take the lag of the time-related features
        self.tabular_feature_df = self.add_time_features_to_df(df=self.tabular_feature_df)
        self.tabular_feature_df = self.add_workhour_labels_to_df(df=self.tabular_feature_df)

        logger.info(f"Final feature set built. Shape: {self.tabular_feature_df.shape}")
        return None