# ⚡ Predictive Paradox

**Electricity Demand Forecasting Pipeline + Interactive Web App**

A production-style machine learning pipeline and web interface built for the IITG.ai recruitment challenge. This project forecasts short-term electricity demand using historical load data, weather signals, and advanced feature engineering—while strictly adhering to classical ML constraints.

---

## 🚀 Project Highlights

* 📊 **End-to-end ML pipeline** (data → features → model → evaluation)
* ⚡ **LightGBM / XGBoost forecasting models**
* 🌦️ Incorporates **weather-driven demand signals**
* 🧠 Advanced **time-series feature engineering (lags, rolling stats, cyclic encoding)**
* 🧹 Robust **anomaly detection & correction**
* 🌐 **Flask-based web interface** for interactive predictions
* 📈 Real-time **visualizations + downloadable results**

---

## 🧩 Problem Context

Electricity demand forecasting is critical for grid stability and cost optimization.
This project focuses on **short-term forecasting (t+1)** using tabular ML models, requiring manual encoding of temporal dependencies. 

---

## 🏗️ Architecture Overview

```
Raw Data → Cleaning → Anomaly Handling → Feature Engineering → Model → Prediction → Visualization
```

---

## 📁 Repository Structure

```
PredictiveParadox/
│
├── dataset/                  # Input + output data
├── graphs/                   # Generated visualizations
├── pipeline/                 # Core ML pipeline modules
│   ├── data.py               # Data loading
│   ├── process.py            # Preprocessing
│   ├── anomaly.py            # Outlier handling
│   ├── feature.py            # Feature engineering
│   ├── model.py              # Model training
│   ├── predictor.py          # Prediction + evaluation
│   └── ...
│
├── main.py                   # CLI entry point
├── app.py                    # Flask web app
├── predictive_paradox_pipeline.ipynb  # Exploration notebook
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd PredictiveParadox

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 (Windows)

pip install -r requirements.txt
```

---

## ▶️ Usage

### Run ML Pipeline (CLI)

```bash
python main.py
```

Default execution:

```python
import pipeline as p

pipe = p.PipeLine1(['dataset/train_demand_data.xlsx', 'dataset/train_weather_data.xlsx'], verbose=True)
pipe.train_model()
pipe.upload(['dataset/test_demand_data.xlsx', 'dataset/test_weather_data.xlsx'])
pipe.predict('prediction1.xlsx')

pipe = p.PipeLine2(['dataset/PGCB_date_power_demand.xlsx', 'dataset/weather_data.xlsx'], verbose=True)
pipe.split(2024)
pipe.train_model()
pipe.predict('prediction2.xlsx')
```

---

### Run Web App

```bash
python app.py
```

Open:
👉 [http://localhost:5000](http://localhost:5000)

---

## 🔬 Machine Learning Pipeline

### 1. Data Processing

* Timestamp parsing & sorting
* Duplicate removal
* Missing/zero-value correction
* Resampling (half-hour → hourly)

### 2. Anomaly Detection

* Rolling window (168 hours)
* Z-score threshold = 2.5
* Replacement with rolling mean

### 3. Feature Engineering

* ⏰ Calendar features (hour, weekday, month)
* 🔁 Cyclical encoding (sin/cos transforms)
* 📉 Lag features (1 → 336 hours)
* 📊 Rolling statistics (mean, std, min, max)
* 🎯 Target: `demand_mw (t+1)`

### 4. Modeling

* LightGBM (default)
* XGBoost (optional)
* Chronological split:

  * Train: pre-2024
  * Test: 2024

### 5. Evaluation

* Metric: **MAPE (Mean Absolute Percentage Error)**
* Feature importance analysis

---

## 📊 Outputs

After execution:

* `dataset/prediction.xlsx` → predictions
* `graphs/anomalous_demand.png` → demand_mw plot with anomalies
* `graphs/clean_demand.png` → cleaned of anomalies
* `graphs/actual_vs_predicted.png` → forecast visualization
* `graphs/feature_importance.png` → model insights

---

## 🌐 Web Application Features

### 🔹 Pipeline I (`/pipeline1`)

* Upload training + multiple test datasets
* Supports batch prediction workflows
* Interactive charts + downloadable outputs

### 🔹 Pipeline II (`/pipeline2`)

* Upload full dataset
* Select train/test split via slider
* Visual evaluation (MAPE, residuals, trends)

---

## 💡 UX Features

* 🔄 Live progress tracking (polling backend jobs)
* 🌀 Animated loading states
* 📦 Drag-and-drop uploads
* 📊 Chart.js visualizations
* 🎨 Dark futuristic UI (inspired by project theme)

---

## ⚡ Backend Design

* Asynchronous execution using **threading**
* Job-based architecture:

```
Request → Job ID → Background Execution → Poll Status → Fetch Results
```

Endpoints:

* `/api/run_pipelineX`
* `/api/job_status/<id>`
* `/api/get_results/<id>`
* `/api/download/<id>`

---

## 🔧 Tech Stack

### ML & Data

* pandas, numpy
* scikit-learn
* lightgbm, xgboost

### Visualization

* matplotlib, seaborn
* Chart.js (frontend)

### Web

* Flask
* HTML/CSS/JS

---

## 📈 Future Improvements

* Multi-horizon forecasting (t+1 → t+k)
* Hyperparameter optimization (Optuna)
* Feature selection / SHAP explainability
* Economic data integration
* Model ensembling

---

## ⚠️ Notes

* Dataset filenames must remain unchanged
* Economic data currently unused in pipeline
* Notebook is exploratory; `main.py` is production entry point

---

## 🎨 Theme Inspiration

Inspired by the **Predictive Paradox concept**:

* Dark, futuristic UI
* Orbital animations
* Data-centric minimalism

---

## 🤝 Contribution

This project was built as part of a recruitment challenge.
Contributions and improvements are welcome!

---