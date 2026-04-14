import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pipeline.data as ld

WINDOW = 168
THRESHOLD = 2.5 

class Anomaly:
    df: ld.Data
    
    def __init__(self, data: ld.Data, verbose = True):
        self.df = data
        if verbose:
            self.display_before()
        self.detect(verbose)
        if verbose:
            self.display_after()
        
    def detect(self, verbose = True):         
        rolling_mean = self.df.df_demand['demand_mw'].rolling(window=WINDOW, center=True, min_periods=1).mean()
        rolling_std  = self.df.df_demand['demand_mw'].rolling(window=WINDOW, center=True, min_periods=1).std()

        z_scores = (self.df.df_demand['demand_mw'] - rolling_mean) / (rolling_std + 1e-8)

        is_anomaly = z_scores.abs() > THRESHOLD
        if verbose:
            print(f'Number of anomalies detected: {is_anomaly.sum()}')

        self.df.df_demand.loc[is_anomaly, 'demand_mw'] = rolling_mean[is_anomaly]

        
    def display_before(self):
        fig = plt.figure(figsize=(15, 4))
        fig.canvas.manager.set_window_title("Anomalous Electricity Demand")
        plt.plot(self.df.df_demand['datetime'], self.df.df_demand['demand_mw'], linewidth=0.5, color='steelblue')
        plt.title('Raw Electricity Demand (MW) — Check for Spikes')
        plt.xlabel('Date')
        plt.ylabel('Demand (MW)')
        plt.tight_layout()
        plt.show()
        
    def display_after(self):
        fig = plt.figure(figsize=(15, 4))
        fig.canvas.manager.set_window_title("Anomaly Free Electricity Demand")
        plt.plot(self.df.df_demand['datetime'], self.df.df_demand['demand_mw'], linewidth=0.5, color='green')
        plt.title('Cleaned Electricity Demand (MW)')
        plt.xlabel('Date')
        plt.ylabel('Demand (MW)')
        plt.tight_layout()
        plt.show()

