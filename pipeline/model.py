import pipeline.data as ld
import lightgbm as lgb

class Model:
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
            verbose= 3 if verbose else -1
        )

        # early_stopping: stop training if validation score doesn't improve for 50 rounds
        self.model.fit(
            self.df.X_train, self.df.y_train
        #    eval_set=[(self.df.X_test, self.df.y_test)],
        #    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        )

        if verbose:
            print('LightGBM training complete!')
    