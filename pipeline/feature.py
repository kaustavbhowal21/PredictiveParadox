import pandas as pd
import numpy as np
import pipeline.data as ld

class Feature:
    df: ld.Data
    
    def __init__(self, data: ld.Data, verbose = True):
        self.df = data
        self.calendar(verbose)
        self.lag(verbose)
        self.rolling(verbose)
        self.set_target(verbose)
    
    def calendar(self, verbose = True):
        self.df.df_merged = self.df.df_merged.sort_values('datetime').reset_index(drop=True)

        self.df.df_merged['hour']        = self.df.df_merged['datetime'].dt.hour          # 0-23
        self.df.df_merged['day_of_week'] = self.df.df_merged['datetime'].dt.dayofweek     # 0=Monday, 6=Sunday
        self.df.df_merged['month']       = self.df.df_merged['datetime'].dt.month         # 1-12
        self.df.df_merged['day_of_year'] = self.df.df_merged['datetime'].dt.dayofyear     # 1-365
        self.df.df_merged['week_of_year']= self.df.df_merged['datetime'].dt.isocalendar().week.astype(int)
        self.df.df_merged['is_weekend']  = (self.df.df_merged['day_of_week'] >= 5).astype(int)  # 1 if Sat/Sun
        self.df.df_merged['quarter']     = self.df.df_merged['datetime'].dt.quarter       # 1-4

        self.df.df_merged['hour_sin']  = np.sin(2 * np.pi * self.df.df_merged['hour'] / 24)
        self.df.df_merged['hour_cos']  = np.cos(2 * np.pi * self.df.df_merged['hour'] / 24)
        self.df.df_merged['month_sin'] = np.sin(2 * np.pi * self.df.df_merged['month'] / 12)
        self.df.df_merged['month_cos'] = np.cos(2 * np.pi * self.df.df_merged['month'] / 12)
        self.df.df_merged['dow_sin']   = np.sin(2 * np.pi * self.df.df_merged['day_of_week'] / 7)
        self.df.df_merged['dow_cos']   = np.cos(2 * np.pi * self.df.df_merged['day_of_week'] / 7)

        if verbose:
            print('Calendar features added.')
            
    def lag(self, verbose = True):
        for lag in [1, 2, 3, 6, 12]:
            self.df.df_merged[f'lag_{lag}h'] = self.df.df_merged['demand_mw'].shift(lag)

        # Same time yesterday and last week
        self.df.df_merged['lag_24h']  = self.df.df_merged['demand_mw'].shift(24)   # 1 day ago, same hour
        self.df.df_merged['lag_48h']  = self.df.df_merged['demand_mw'].shift(48)   # 2 days ago
        self.df.df_merged['lag_168h'] = self.df.df_merged['demand_mw'].shift(168)  # 7 days ago, same hour (most powerful!)
        self.df.df_merged['lag_336h'] = self.df.df_merged['demand_mw'].shift(336)  # 14 days ago

        if verbose:
            print('Lag features added.')
            
    def rolling(self, verbose = True):
        demand_shifted = self.df.df_merged['demand_mw'].shift(1)  # start from t-1

        for window in [3, 6, 12, 24, 168]:
            self.df.df_merged[f'roll_mean_{window}h'] = demand_shifted.rolling(window=window, min_periods=1).mean()
            self.df.df_merged[f'roll_std_{window}h']  = demand_shifted.rolling(window=window, min_periods=1).std()

        self.df.df_merged['roll_min_24h'] = demand_shifted.rolling(window=24, min_periods=1).min()
        self.df.df_merged['roll_max_24h'] = demand_shifted.rolling(window=24, min_periods=1).max()

        if verbose:
            print('Rolling features added.')
            
    def set_target(self, verbose = True):
        self.df.df_merged['target_demand_mw'] = self.df.df_merged['demand_mw'].shift(-1)

        self.df.df_merged = self.df.df_merged.dropna(subset=['target_demand_mw', 'lag_168h'])

        if verbose:
            print(f'Final dataset shape after feature engineering: {self.df.df_merged.shape}')
            print(self.df.df_merged.head())