import pipeline.data as ld
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from joblib import parallel_backend

param_one = {
    "n_estimators": [1000],
    "learning_rate": [0.05],
    "num_leaves": [63],
    "min_child_samples": [20],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

param_dist = {
    "n_estimators": [200, 500, 1000, 1500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [-1, 5, 10, 15],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [0, 0.1, 1]
}

class LightGBM:
    train_df: ld.Data
    model: lgb.LGBMRegressor
    
    def __init__(self, data: ld.Data, verbose = True):
        self.train_df = data
        self.train(verbose)
        
    def train(self, verbose = True):
        # estimator
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # self.model = RandomizedSearchCV(
        #     self.model,
        #     param_distributions=param_dist,
        #     n_iter=10,     
        #     scoring="neg_mean_absolute_percentage_error",
        #     cv=3,
        #     verbose=0,
        #     n_jobs=-1
        # )

        # train model
        with parallel_backend("threading"):
            self.model.fit(self.train_df.df_finalX, self.train_df.df_finalY)

        # if verbose:
        #     print('Best Parameters:')
        #     print(self.model.best_params_)
            
        # self.model = self.model.best_estimator_

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
