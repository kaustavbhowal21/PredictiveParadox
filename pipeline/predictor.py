import pipeline.data as ld
import pipeline.model as ml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error

class Predictor:
    df: ld.Data
    model: ml.Model
    y_pred: pd.Series
    mape = 0.0
    
    def __init__(self, data, model, verbose = True):
        self.df = data
        self.model = model
        self.predict(verbose)
        
    def display(self, verbose = True):
        fig, ax = plt.subplots(figsize=(15, 5))
        fig.canvas.manager.set_window_title("Actual vs Predicted")

        start = 0
        end = -1
        ax.plot(self.df.test_df['datetime'].values[start:end], self.df.y_test.values[start:end],
                label='Actual', color='steelblue', linewidth=1.2)
        ax.plot(self.df.test_df['datetime'].values[start:end], self.y_pred[start:end],
                label='Predicted', color='tomato', linewidth=1.2, linestyle='--')

        ax.set_title(f'Actual vs Predicted Demand — Month 1\nMAPE: {self.mape:.2f}%')
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand (MW)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('graphs/actual_vs_predicted.png', dpi=150)
        if verbose:
            plt.show()
        
    def evaluate(self, verbose = True):
        
        importance_df = pd.DataFrame({
            'feature': self.df.feature_cols,
            'importance': self.model.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Plot top 25 features
        top_n = 25
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.canvas.manager.set_window_title("Feature Importances")
        sns.barplot(
            data=importance_df.head(top_n),
            x='importance', y='feature',
            palette='viridis', ax=ax
        )
        ax.set_title(f'Top {top_n} Feature Importances (LightGBM)')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        plt.savefig('graphs/feature_importance.png', dpi=150)
        if verbose:
            plt.show()

            print('Top 10 most important features:')
            print(importance_df.head(10).to_string(index=False))
        
    def predict(self, verbose = True):
        self.y_pred = self.model.model.predict(self.df.X_test)
        
        # Calculate MAPE
        # MAPE = average of |actual - predicted| / actual * 100%
        self.mape = mean_absolute_percentage_error(self.df.y_test, self.y_pred) * 100
        if verbose:
            print(f'\n=== TEST MAPE: {self.mape:.2f}% ===')
            print()
            if self.mape < 5:
                print('Excellent! Less than 5% error is industry-grade.')
            elif self.mape < 10:
                print('Good. Less than 10% is acceptable for short-term forecasting.')
            else:
                print('Room for improvement. Consider more lag features or hyperparameter tuning.')
            
        self.display(verbose)
        self.evaluate(verbose)