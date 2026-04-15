# Predictive Paradox

Electricity demand forecasting pipeline built for the IITG.ai recruitment task. The project combines historical power demand data, weather observations, anomaly handling, feature engineering, and gradient-boosted regressors to predict the next-hour electricity demand.

## Overview

The pipeline:

1. Loads raw demand and weather files from the `dataset/` directory.
2. Cleans timestamps, removes duplicate records, and repairs missing or zero-valued generation mix fields.
3. Smooths half-hour demand readings into hourly demand values.
4. Detects and replaces demand anomalies using a rolling z-score approach.
5. Merges cleaned demand data with weather data.
6. Builds calendar, lag, and rolling-window features.
7. Trains a regressor on pre-2024 data and evaluates on 2024 data.
8. Exports prediction results and visualization artifacts.

## Repository Structure

```text
PredictiveParadox/
|-- dataset/
|   |-- PGCB_date_power_demand.xlsx
|   |-- weather_data.xlsx
|   |-- economic_full_1.csv
|   `-- prediction.xlsx
|-- graphs/
|   |-- actual_vs_predicted.png
|   `-- feature_importance.png
|-- pipeline/
|   |-- __init__.py
|   |-- anomaly.py
|   |-- data.py
|   |-- feature.py
|   |-- model.py
|   |-- predictor.py
|   `-- process.py
|-- main.py
|-- predictive_paradox_pipeline.ipynb
|-- requirements.txt
`-- README.md
```

## Data Requirements

The code expects the following files inside `dataset/` with these exact names:

- `PGCB_date_power_demand.xlsx`: historical electricity generation and demand data
- `weather_data.xlsx`: weather metadata and hourly weather observations

`economic_full_1.csv` is present in the repository, but it is not currently consumed by the Python pipeline.

## How It Works

### 1. Data Loading

`pipeline.data.Data` loads:

- demand data from `dataset/PGCB_date_power_demand.xlsx`
- location metadata from the first two rows of `dataset/weather_data.xlsx`
- weather observations from the remaining rows of `dataset/weather_data.xlsx`

### 2. Preprocessing

`pipeline.process.DataProcessor` performs:

- datetime parsing and sorting
- duplicate timestamp removal
- zero and missing generation-mix repair using nearby valid rows
- resampling from half-hour intervals to hourly intervals

### 3. Anomaly Handling

`pipeline.anomaly.Anomaly` identifies outliers in `demand_mw` using a centered rolling window of 168 hours and a z-score threshold of 2.5, then replaces anomalous values with the rolling mean.

### 4. Feature Engineering

`pipeline.feature.Feature` adds:

- calendar features such as hour, month, day of week, quarter, and weekend flag
- cyclical encodings using sine and cosine transforms
- lag features from 1 hour to 336 hours
- rolling statistics such as mean, standard deviation, minimum, and maximum
- `target_demand_mw` as the next-hour prediction target

### 5. Modeling

The dataset is split by year:

- training set: all rows before 2024
- test set: rows from 2024

Supported regressors:

- `LGBR`: LightGBM regressor
- `XGBR`: XGBoost regressor

The default example in `main.py` uses LightGBM.

### 6. Evaluation and Outputs

`pipeline.predictor.Predictor`:

- predicts on the 2024 holdout set
- computes MAPE
- saves an actual-vs-predicted plot to `graphs/actual_vs_predicted.png`
- saves feature importance to `graphs/feature_importance.png`
- exports the final prediction workbook to `dataset/prediction.xlsx`

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run the default pipeline:

```powershell
python main.py
```

`main.py` currently runs:

```python
import pipeline as p

pipe = p.PipeLine('dataset/', True)
pipe.process()
pipe.predict('prediction.xlsx', regressor='LGBR')
```

To switch the model, change the regressor argument to `XGBR`.

## Output Files

After a successful run, you should expect:

- `dataset/prediction.xlsx`: actual and predicted demand values for the test period
- `graphs/actual_vs_predicted.png`: demand forecast comparison plot
- `graphs/feature_importance.png`: feature ranking plot

## Key Dependencies

Major libraries used in this project:

- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `matplotlib`
- `seaborn`
- `openpyxl`

## Notes

- The pipeline assumes the dataset filenames and folder structure remain unchanged.
- Prediction output is saved inside `dataset/`, even though only the filename is passed to `pipe.predict(...)`.
- Visualization display depends on the `verbose` flag. Image files are still written to the `graphs/` directory.
- The notebook `predictive_paradox_pipeline.ipynb` appears to be the exploratory or development version of the workflow, while `main.py` is the runnable script entry point.

---

## Flask Setup

```bash
# Run the app
python app.py
# → Open http://localhost:5000
```

---

## Features

### Pipeline I (`/pipeline1`)
- Upload **training demand** + **training weather** files
- Upload optional **economic CSV** (passed to pipeline if present)
- Add **multiple test dataset tabs** (Dataset 1, Dataset 2, ...) for the same trained model
- Each tab has its own demand + weather upload zones
- On submit: calls `PipeLine1(train_files).train_model()` → `.upload(test_files)` → `.predict()`
- Results page: metrics, interactive Chart.js line chart, data preview table, download button

### Pipeline II (`/pipeline2`)
- Upload **PGCB power demand** + **weather** files
- Optional **economic CSV**
- Interactive **year-split slider** (2019–2026) with visual train/test proportion bar
- On submit: calls `PipeLine2(data_files).split(year).train_model().predict()`
- Results page: metrics including MAPE, actual vs predicted chart, residuals bar chart, download

### UX
- **Loading overlay** with animated orbital rings while ML model trains
- Live **progress messages** polled every 1.5 seconds from backend
- **Toast notifications** for errors and completion
- **Drag-and-drop** file upload zones
- Fully responsive dark/white shiny theme matching the problem statement aesthetic

---

## How Jobs Work

The backend uses Python `threading` + an in-memory `jobs` dict:
1. POST to `/api/run_pipeline1` or `/api/run_pipeline2` → saves files, starts thread, returns `job_id`
2. Frontend polls `/api/job_status/<job_id>` every 1.5s
3. When status = `"done"`, frontend fetches `/api/get_results/<job_id>` for chart data
4. Download via `/api/download/<job_id>`

---

## Extending for Multi-Tab Predictions

The backend currently passes the **first test tab** to `PipeLine1`. To support multiple test tabs predicting independently, extend `run_pipeline1_job` to loop over `test_file_groups` and call `pipe.upload(...)` + `pipe.predict(...)` for each group, storing multiple output paths in `jobs[job_id]['outputs']`.

---

## Theme

Inspired by the Predictive Paradox PDF cover:
- Pure black background with starfield and perspective grid floor
- **Orbitron** display font, **Rajdhani** body, **Space Mono** for data/labels  
- White-on-black with glowing borders and subtle radial gradients
- Orbital animation for loading screen