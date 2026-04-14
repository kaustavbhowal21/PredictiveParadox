import numpy as np
import bisect
import pandas as pd
import pipeline.data as ld

LEN = 24
        
class DataProcessor:
    cols = ['generation_mw', 'demand_mw', 'load_shedding', 'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']
    df: ld.Data
    
    def __init__(self, data: ld.Data, verbose = True):
        self.df = data
        self.process_date(verbose)
        self.remove_duplicates(verbose)
        self.generalize(verbose)
        self.manage_half_hours(verbose)
        
    def k_nearest(self, L, x, k):
        pos = bisect.bisect_left(L, x)
        
        left = pos - 1
        right = pos
        
        result = []
        
        while k > 0 and (left >= 0 or right < len(L)):
            if left < 0:
                result.append(L[right])
                right += 1
            elif right >= len(L):
                result.append(L[left])
                left -= 1
            else:
                if abs(L[left] - x) <= abs(L[right] - x):
                    result.append(L[left])
                    left -= 1
                else:
                    result.append(L[right])
                    right += 1
            k -= 1
        
        return result
        
    def process_date(self, verbose = True):
        self.df.df_demand['datetime'] = pd.to_datetime(self.df.df_demand['datetime'])  
        self.df.df_weather['datetime'] = pd.to_datetime(self.df.df_weather['time'])  
        self.df.df_weather = self.df.df_weather.drop('time', axis=1)
        self.df.df_demand[self.cols] = self.df.df_demand[self.cols].fillna(0)

        self.df.df_demand  = self.df.df_demand.sort_values('datetime').reset_index(drop=True)
        self.df.df_weather = self.df.df_weather.sort_values('datetime').reset_index(drop=True)

        if verbose:
            print('Demand date range:', self.df.df_demand['datetime'].min(), 'to', self.df.df_demand['datetime'].max())
            print('Weather date range:', self.df.df_weather['datetime'].min(), 'to', self.df.df_weather['datetime'].max())
            
            print('Demand columns:', self.df.df_demand.columns.tolist())
            print('Location columns:', self.df.df_locate.columns.tolist())
            print('Weather columns:', self.df.df_weather.columns.tolist())
            
    def remove_duplicates(self, verbose = True):
        n_dupes = self.df.df_demand.duplicated(subset='datetime').sum()
        if verbose:
            print(f'Number of duplicate timestamps in demand data: {n_dupes}')

        self.df.df_demand = self.df.df_demand.drop_duplicates(subset='datetime', keep='first')
        if verbose:
            print(f'Demand rows after removing duplicates: {len(self.df.df_demand)}')
            
    def generalize(self, verbose = True):
        self.df.df_demand[self.cols] = self.df.df_demand[self.cols].astype(float)

        filtered_indices = self.df.df_demand.index[(self.df.df_demand[self.cols].notna() & (self.df.df_demand[self.cols] != 0)).any(axis=1)]
        masked = self.df.df_demand.loc[(self.df.df_demand[self.cols].isna() | (self.df.df_demand[self.cols] == 0)).all(axis=1)]

        for idx, _ in masked.iterrows():
            valid_indices = self.k_nearest(filtered_indices, idx, LEN)

            valid_rows = self.df.df_demand.loc[valid_indices, ['generation_mw'] + self.cols]
            for col in self.cols:
                valid_rows[col] = valid_rows[col] / valid_rows['generation_mw']
            valid_means = valid_rows[self.cols].mean()
            sum = 0.0
            for col in self.cols:
                sum += valid_means[col]
            valid_means = valid_means / sum

            self.df.df_demand.loc[idx, self.cols] = self.df.df_demand.loc[idx, 'generation_mw'] * valid_means
        
        if verbose:
            print(self.df.df_demand.head())
    
    def custom_avg(self, x : np.array):
        if x.size == 2:
            if np.isnan(x[0]):
                return x[1]
            elif np.isnan(x[1]):
                return x[0]
            else:
                return x.mean()
        if np.isnan(x[0]) and np.isnan(x[2]):
            return x[1]
        elif np.isnan(x[0]):
            return (3 * x[1] + x[2]) / 4
        else:
            return (2 * x[1] + x[0] + x[2]) / 4
    
    def manage_half_hours(self, verbose = True):
        self.df.df_demand = self.df.df_demand.set_index('datetime')
        cols = ['generation_mw', 'demand_mw']

        if verbose:
            print(f'Demand rows before processing half hourly values : {len(self.df.df_demand)}')

        self.df.df_demand = self.df.df_demand.resample('30min').asfreq()

        self.df.df_demand[cols] = self.df.df_demand[cols].rolling('90min', center=True, min_periods=1).apply(self.custom_avg, raw=True)
        self.df.df_demand = self.df.df_demand.resample('h').first()

        self.df.df_demand[self.cols] = self.df.df_demand[self.cols].interpolate(method='linear', limit=6)
        self.df.df_demand = self.df.df_demand.dropna(subset=self.cols, how='all')

        self.df.df_demand = self.df.df_demand.reset_index()

        if verbose:
            print(f'Demand rows after processing half hourly values : {len(self.df.df_demand)}')        