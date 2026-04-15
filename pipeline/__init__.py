VERSION = "1.0.1"
AUTHOR = "Kaustav Bhowal"

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pipeline.data import Data
from pipeline.process import DataProcessor
from pipeline.anomaly import Anomaly
from pipeline.feature import Feature
from pipeline.model import LightGBM, XGBoost
from pipeline.predictor import Predictor

feature_cols = ['load_shedding', 'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal', 'temperature_2m (°C)', 'relative_humidity_2m (%)', 'apparent_temperature (°C)', 'precipitation (mm)', 'dew_point_2m (°C)', 'soil_temperature_0_to_7cm (°C)', 'wind_direction_10m (°)', 'cloud_cover (%)', 'sunshine_duration (s)', 'hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year', 'is_weekend', 'quarter', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_48h', 'lag_168h', 'lag_336h', 'roll_mean_3h', 'roll_std_3h', 'roll_mean_6h', 'roll_std_6h', 'roll_mean_12h', 'roll_std_12h', 'roll_mean_24h', 'roll_std_24h', 'roll_mean_168h', 'roll_std_168h', 'roll_min_24h', 'roll_max_24h']

class PipeLine1:
    train: Data
    test: Data
    model = None
    predictor: Predictor
    verbose: bool
    prediction: pd.DataFrame
    
    def __init__(self, files, verbose = True):
        print("Processing Train Data...")
        self.verbose = verbose
        self.train = Data(files, verbose=self.verbose)  
        
        DataProcessor(self.train, self.verbose)
        Anomaly(self.train, self.verbose)
        self.train.merge(verbose=self.verbose)
        Feature(self.train, self.verbose)      
        self.train.prepare(feature_cols, 'target_demand_mw', self.verbose)
        
    def upload(self, test_files: str):
        print('Processing Test Data...')
        self.test = Data(test_files, verbose=self.verbose)
        
        train_demand = self.train.df_demand
        train_weather = self.train.df_weather
        
        self.test.df_demand['flag'] = 1
        self.test.df_weather['flag'] = 1
        train_demand['flag'] = 0
        train_weather['flag'] = 0
        
        
        self.test.df_demand = pd.concat([self.train.df_demand, self.test.df_demand])
        self.test.df_weather = pd.concat([self.train.df_weather, self.test.df_weather])        
        
        DataProcessor(self.test, self.verbose)
        Anomaly(self.test, self.verbose)
        self.test.merge(cols=['datetime', 'flag'], verbose=self.verbose)
        Feature(self.test, self.verbose)
        
        self.test.df_demand = self.test.df_demand[self.test.df_demand['flag'] == 1]
        self.test.df_weather = self.test.df_weather[self.test.df_weather['flag'] == 1]
        self.test.df_merged = self.test.df_merged[self.test.df_merged['flag'] == 1]
        
        self.test.df_demand.drop('flag', axis=1)
        self.test.df_weather.drop('flag', axis=1)
        self.test.df_merged.drop('flag', axis=1)
        
        print(self.test.df_merged)
        
        self.test.prepare(feature_cols, 'target_demand_mw', self.verbose)
        
    def train_model(self):
        print('Training Model...')
        self.model = LightGBM(self.train, self.verbose)
        
    def predict(self, prediction_file: str):
        print("Predicting...")
        if self.verbose:
            print(f'Training rows: {len(self.train.df_finalX)}')
            print(f'Test rows:     {len(self.test.df_finalX)}')
            print(f'Train period: {self.train.df_merged["datetime"].min()} to {self.train.df_merged["datetime"].max()}')
            print(f'Test  period: {self.test.df_merged["datetime"].min()} to {self.test.df_merged["datetime"].max()}')
        self.predictor = Predictor(self.test, self.model, self.verbose)
        
        print("Saving Prediction... ")
        self.prediction = self.test.df_merged[['datetime', 'generation_mw', 'load_shedding', 'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']].shift(-1)
        
        self.prediction['actual_demand_mw'] = self.test.df_finalY
        self.prediction['predicted_demand_mw'] = self.predictor.y_pred
        self.prediction = self.prediction.set_index('datetime')
        self.prediction.to_excel(prediction_file)
        

class PipeLine2:
    data: Data
    test: Data
    train: Data
    model = None
    predictor: Predictor
    verbose: bool
    prediction: pd.DataFrame
    
    def __init__(self, files, verbose = True):
        print("Processing Data...")
        self.verbose = verbose
        self.data = Data(files, verbose=self.verbose)  
        
        DataProcessor(self.data, self.verbose)
        Anomaly(self.data, self.verbose)
        self.data.merge(verbose=self.verbose)
        Feature(self.data, self.verbose)     
        
    def split(self, year = 2024):
        print('Splitting Data...')
        
        self.train = Data(None, self.data.df_demand[self.data.df_demand['datetime'].dt.year < year], self.data.df_weather[self.data.df_weather['datetime'].dt.year < year], self.data.df_merged[self.data.df_merged['datetime'].dt.year < year])
        self.test = Data(None, self.data.df_demand[self.data.df_demand['datetime'].dt.year >= year], self.data.df_weather[self.data.df_weather['datetime'].dt.year >= year], self.data.df_merged[self.data.df_merged['datetime'].dt.year >= year])
                 
        self.train.prepare(feature_cols, 'target_demand_mw', self.verbose)
        self.test.prepare(feature_cols, 'target_demand_mw', self.verbose)
        
    def train_model(self):
        print('Training Model...')
        self.model = LightGBM(self.train, self.verbose)
        
    def predict(self, prediction_file: str):
        print("Predicting...")
        if self.verbose:
            print(f'Training rows: {len(self.train.df_finalX)}')
            print(f'Test rows:     {len(self.test.df_finalX)}')
            print(f'Train period: {self.train.df_merged["datetime"].min()} to {self.train.df_merged["datetime"].max()}')
            print(f'Test  period: {self.test.df_merged["datetime"].min()} to {self.test.df_merged["datetime"].max()}')
        self.predictor = Predictor(self.test, self.model, self.verbose)
        
        print("Saving Prediction... ")
        self.prediction = self.test.df_merged[['datetime', 'generation_mw', 'load_shedding', 'gas', 'liquid_fuel', 'coal', 'hydro', 'solar', 'wind', 'india_bheramara_hvdc', 'india_tripura', 'india_adani', 'nepal']].shift(-1)
        
        self.prediction['actual_demand_mw'] = self.test.df_finalY
        self.prediction['predicted_demand_mw'] = self.predictor.y_pred
        self.prediction = self.prediction.set_index('datetime')
        self.prediction.to_excel(prediction_file)