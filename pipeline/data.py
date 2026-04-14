import pandas as pd     

class Data:
    folder: str
    feature_cols = []
    
    df_demand: pd.DataFrame
    df_locate: pd.DataFrame
    df_locate: pd.DataFrame
    df_merged: pd.DataFrame
    
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    
    final: pd.DataFrame
    
    def __init__(self, folder: str, verbose = True):
        self.folder = folder
        self.load()
        if verbose:
            self.display()
    
    def display(self):
        print('Demand shape:', self.df_demand.shape)
        print('Location shape:', self.df_locate.shape)
        print('Weather shape:', self.df_weather.shape)
        
        print('=== DEMAND DATA ===')
        print(self.df_demand.head())

        print('=== LOCATION DATA ===')
        print(self.df_locate.head())

        print('=== WEATHER DATA ===')
        print(self.df_weather.head())
                
        print('=== DEMAND INFO ===')
        self.df_demand.info()
        print()
        print('=== LOCATION INFO ===')
        self.df_locate.info()
        print()
        print('=== WEATHER INFO ===')
        self.df_weather.info()
    
    def load(self):
        self.df_demand  = pd.read_excel(self.folder + 'PGCB_date_power_demand.xlsx')
        self.df_locate = pd.read_excel(self.folder + 'weather_data.xlsx', nrows=2)
        self.df_weather = pd.read_excel(self.folder + 'weather_data.xlsx', skiprows=3)
        
    def merge(self, verbose = True):
        self.df_merged = self.df_demand.merge(self.df_weather, on='datetime', how='left')

        if verbose:
            print('Combined dataset shape:', self.df_merged.shape)
            print(self.df_merged.head())
            print('Missing values in combined data:')
            print(self.df_merged.isnull().sum())

        weather_cols = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'apparent_temperature (°C)', 'precipitation (mm)', 'dew_point_2m (°C)', 'soil_temperature_0_to_7cm (°C)', 'wind_direction_10m (°)', 'cloud_cover (%)', 'sunshine_duration (s)']  # <-- replace with real names
        self.df_merged[weather_cols] = self.df_merged[weather_cols].ffill().bfill()
        
        # self.df_merged.to_excel(self.folder + 'merged.xlsx')
        
    def split(self, verbose = True):
        exclude_cols = ['datetime', 'demand_mw', 'target_demand_mw', 'year', 'remarks']
        self.feature_cols = [c for c in self.df_merged.columns if c not in exclude_cols]

        if verbose:
            print(f'Number of features: {len(self.feature_cols)}')
            print('Features:', self.feature_cols)
            
        self.train_df = self.df_merged[self.df_merged['datetime'].dt.year < 2024].copy()
        self.test_df  = self.df_merged[self.df_merged['datetime'].dt.year == 2024].copy()

        self.X_train = self.train_df[self.feature_cols]
        self.y_train = self.train_df['target_demand_mw']

        self.X_test  = self.test_df[self.feature_cols]
        self.y_test  = self.test_df['target_demand_mw']

        if verbose:
            print(f'Training rows: {len(self.X_train)}')
            print(f'Test rows:     {len(self.X_test)}')
            print(f'Train period: {self.train_df["datetime"].min()} to {self.train_df["datetime"].max()}')
            print(f'Test  period: {self.test_df["datetime"].min()} to {self.test_df["datetime"].max()}')
        
    def save(self, file: str, pred):
        self.final = self.test_df[['datetime', 'generation_mw', 'load_shedding', 'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']].shift(-1)
        
        self.final['actual_demand_mw'] = self.y_test
        self.final['predicted_demand_mw'] = pred
        self.final = self.final.set_index('datetime')
        self.final.to_excel(self.folder + file)