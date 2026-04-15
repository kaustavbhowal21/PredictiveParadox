import pipeline.data as ld
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LightGBM:
    df: ld.Data
    model: lgb.LGBMRegressor
    
    def __init__(self, data: ld.Data, verbose = True):
        self.df = data
        self.train(verbose)
        
    def train(self, verbose = True):
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,       # number of trees
            learning_rate=0.05,      # how much each tree contributes
            num_leaves=63,           # complexity of each tree
            min_child_samples=20,    # minimum data in leaf (prevents overfitting)
            subsample=0.8,           # use 80% of data per tree (prevents overfitting)
            colsample_bytree=0.8,    # use 80% of features per tree
            random_state=42,
            verbose= 1 if verbose else -1
        )

        # train model
        self.model.fit(self.df.X_train, self.df.y_train)

        if verbose:
            print('LightGBM training complete!')


class XGBoost:
    df: ld.Data
    model: xgb.XGBRegressor
    
    def __init__(self, data: ld.Data, verbose = True):
        self.df = data
        self.train(verbose)
        
    def train(self, verbose = True):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,       # number of trees
            learning_rate=0.05,      # how much each tree contributes
            num_leaves=63,           # complexity of each tree
            min_child_samples=20,    # minimum data in leaf (prevents overfitting)
            subsample=0.8,           # use 80% of data per tree (prevents overfitting)
            colsample_bytree=0.8,    # use 80% of features per tree
            random_state=42,
            verbose= 1 if verbose else -1
        )

        # train model
        self.model.fit(self.df.X_train, self.df.y_train)

        if verbose:
            print('XGBoost training complete!')
