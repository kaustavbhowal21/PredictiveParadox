VERSION = "1.0.1"
AUTHOR = "Kaustav Bhowal"

import warnings
warnings.filterwarnings('ignore')
from pipeline.data import Data
from pipeline.process import DataProcessor
from pipeline.anomaly import Anomaly
from pipeline.feature import Feature
from pipeline.model import LightGBM, XGBoost
from pipeline.predictor import Predictor

class PipeLine:
    data: Data
    model = None
    predictor: Predictor
    verbose: bool
    
    def __init__(self, folder: str, verbose = True):
        self.verbose = verbose
        print("Loading Folder... ", folder)
        self.data = Data(folder, verbose)
        
    def process(self):
        print("Processing...")
        DataProcessor(self.data, self.verbose)
        Anomaly(self.data, self.verbose)
        self.data.merge(self.verbose)
        Feature(self.data, self.verbose)
        
    def predict(self, file: str, regressor: str):
        print("Predicting...")
        self.data.split(self.verbose)
        if regressor == 'XGBR':
            self.model = XGBoost(self.data, self.verbose)
        elif regressor == 'LGBR':
            self.model = LightGBM(self.data, self.verbose)
        self.predictor = Predictor(self.data, self.model, self.verbose)
        print("Saving Prediction... ", file)
        self.data.save(file, self.predictor.y_pred)
    