import pandas as pd     

class Data:
    files: list
    feature_cols = []
    
    df_demand: pd.DataFrame
    df_locate: pd.DataFrame
    df_weather: pd.DataFrame
    
    df_merged: pd.DataFrame    
    df_finalX: pd.DataFrame
    df_finalY: pd.DataFrame
    
    def __init__(self, files = None, df_demand = None, df_weather = None, df_merged = None, verbose = True):
        if df_demand is not None:
            self.df_demand = df_demand.copy()
        if df_weather is not None:
            self.df_weather = df_weather.copy()
        if df_merged is not None:
            self.df_merged = df_merged.copy()
        if files is not None:
            self.files = files
            self.load()
            if verbose:
                self.display()
    
    def display(self):
        print('Demand shape:', self.df_demand.shape)
        # print('Location shape:', self.df_locate.shape)
        print('Weather shape:', self.df_weather.shape)
        
        print('=== DEMAND DATA ===')
        print(self.df_demand.head())

        # print('=== LOCATION DATA ===')
        # print(self.df_locate.head())

        print('=== WEATHER DATA ===')
        print(self.df_weather.head())
                
        print('=== DEMAND INFO ===')
        self.df_demand.info()
        print()
        # print('=== LOCATION INFO ===')
        # self.df_locate.info()
        # print()
        print('=== WEATHER INFO ===')
        self.df_weather.info()
    
    def load(self):
        self.df_demand  = pd.read_excel(self.files[0])
        df_locate = pd.read_excel(self.files[1], nrows=2)
        if 'latitude' in df_locate.columns:
            self.df_weather = pd.read_excel(self.files[1], skiprows=3)
        else:
            self.df_weather = pd.read_excel(self.files[1])
        
    def merge(self, cols = ['datetime'], verbose = True):
        self.df_merged = self.df_demand.merge(self.df_weather, on=cols, how='left')

        if verbose:
            print('Combined dataset shape:', self.df_merged.shape)
            print(self.df_merged.head())
            print('Missing values in combined data:')
            print(self.df_merged.isnull().sum())

        weather_cols = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'apparent_temperature (°C)', 'precipitation (mm)', 'dew_point_2m (°C)', 'soil_temperature_0_to_7cm (°C)', 'wind_direction_10m (°)', 'cloud_cover (%)', 'sunshine_duration (s)']  # <-- replace with real names
        self.df_merged[weather_cols] = self.df_merged[weather_cols].ffill().bfill()
        
    def prepare(self, xcols, ycol, verbose = True):
        self.feature_cols = xcols
        if verbose:
            print(f'Number of features: {len(xcols)}')
            print('Features:', xcols)
            
        self.df_finalX = self.df_merged[xcols].copy()
        self.df_finalY = self.df_merged[ycol].copy()