from flask import Flask, render_template, request, jsonify, send_file, session
import os
import threading
import uuid
import json
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'predictive_paradox_secret_key_2024'

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Job store: job_id -> {status, result, error, progress_msg}
jobs = {}
jobs_lock = threading.Lock()

ALLOWED_EXT = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def run_pipeline1_job(job_id, train_files, test_files, output_path):
    try:
        with jobs_lock:
            jobs[job_id]['progress'] = 'Loading pipeline...'
        import pipeline as p
        with jobs_lock:
            jobs[job_id]['progress'] = 'Training model on uploaded data...'
        pipe = p.PipeLine1(train_files, verbose=False)
        pipe.train_model()
        with jobs_lock:
            jobs[job_id]['progress'] = 'Generating predictions...'
        pipe.upload(test_files)
        pipe.predict(output_path)
        with jobs_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['progress'] = 'Complete'
            jobs[job_id]['output'] = output_path
    except Exception as e:
        with jobs_lock:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['traceback'] = traceback.format_exc()

def run_pipeline2_job(job_id, data_files, split_year, output_path):
    try:
        with jobs_lock:
            jobs[job_id]['progress'] = 'Loading pipeline...'
        import pipeline as p
        with jobs_lock:
            jobs[job_id]['progress'] = 'Splitting data and training model...'
        pipe = p.PipeLine2(data_files, verbose=False)
        pipe.split(split_year)
        pipe.train_model()
        with jobs_lock:
            jobs[job_id]['progress'] = 'Generating predictions...'
        pipe.predict(output_path)
        with jobs_lock:
            jobs[job_id]['status'] = 'done'
            jobs[job_id]['progress'] = 'Complete'
            jobs[job_id]['output'] = output_path
    except Exception as e:
        with jobs_lock:
            jobs[job_id]['status'] = 'error'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['traceback'] = traceback.format_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pipeline1')
def pipeline1():
    return render_template('pipeline1.html')

@app.route('/pipeline2')
def pipeline2():
    return render_template('pipeline2.html')

@app.route('/api/run_pipeline1', methods=['POST'])
def api_run_pipeline1():
    try:
        train_demand = request.files.get('train_demand')
        train_weather = request.files.get('train_weather')
        economic = request.files.get('economic')  # optional

        test_tabs = []
        i = 0
        while True:
            td = request.files.get(f'test_demand_{i}')
            tw = request.files.get(f'test_weather_{i}')
            if td is None and tw is None:
                break
            if td:
                test_tabs.append({'demand': td, 'weather': tw})
            i += 1

        if not train_demand or not train_weather:
            return jsonify({'error': 'Training files required'}), 400
        if not test_tabs:
            return jsonify({'error': 'At least one test dataset required'}), 400

        job_id = str(uuid.uuid4())

        # Save train files
        train_demand_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_train_demand.xlsx')
        train_weather_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_train_weather.xlsx')
        train_demand.save(train_demand_path)
        train_weather.save(train_weather_path)

        train_files = [train_demand_path, train_weather_path]

        if economic and allowed_file(economic.filename):
            eco_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_economic.csv')
            economic.save(eco_path)
            train_files.append(eco_path)

        # Save test files
        all_test_results = []
        test_file_groups = []
        for idx, tab in enumerate(test_tabs):
            td_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_test_demand_{idx}.xlsx')
            tw_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_test_weather_{idx}.xlsx')
            tab['demand'].save(td_path)
            if tab['weather']:
                tab['weather'].save(tw_path)
                test_file_groups.append([td_path, tw_path])
            else:
                test_file_groups.append([td_path])

        output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_prediction1.xlsx')

        with jobs_lock:
            jobs[job_id] = {'status': 'running', 'progress': 'Initializing...', 'pipeline': 1,
                            'test_count': len(test_tabs), 'output': None, 'error': None}

        # Pass first test group (multi-tab support can be extended in pipeline)
        test_files_flat = test_file_groups[0] if test_file_groups else []

        t = threading.Thread(target=run_pipeline1_job, args=(job_id, train_files, test_files_flat, output_path))
        t.daemon = True
        t.start()

        return jsonify({'job_id': job_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_pipeline2', methods=['POST'])
def api_run_pipeline2():
    try:
        pgcb_file = request.files.get('pgcb_data')
        weather_file = request.files.get('weather_data')
        economic = request.files.get('economic')
        split_year = request.form.get('split_year', 2024)

        if not pgcb_file or not weather_file:
            return jsonify({'error': 'PGCB and Weather files required'}), 400

        job_id = str(uuid.uuid4())

        pgcb_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_pgcb.xlsx')
        weather_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_weather.xlsx')
        pgcb_file.save(pgcb_path)
        weather_file.save(weather_path)

        data_files = [pgcb_path, weather_path]

        if economic and allowed_file(economic.filename):
            eco_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_economic.csv')
            economic.save(eco_path)
            data_files.append(eco_path)

        output_path = os.path.join(OUTPUT_FOLDER, f'{job_id}_prediction2.xlsx')

        with jobs_lock:
            jobs[job_id] = {'status': 'running', 'progress': 'Initializing...', 'pipeline': 2,
                            'output': None, 'error': None}

        t = threading.Thread(target=run_pipeline2_job, args=(job_id, data_files, int(split_year), output_path))
        t.daemon = True
        t.start()

        return jsonify({'job_id': job_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job_status/<job_id>')
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job['status'],
        'progress': job.get('progress', ''),
        'error': job.get('error'),
        'pipeline': job.get('pipeline')
    })

@app.route('/api/get_results/<job_id>')
def get_results(job_id):
    """Return prediction data as JSON for charting."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Results not ready'}), 404

    output_path = job['output']
    try:
        import pandas as pd
        df = pd.read_excel(output_path)
        # Return first 500 rows for charting
        preview = df.head(len(df))
        cols = list(preview.columns)
        data = preview.values.tolist()
        # Try to find datetime and demand columns
        datetime_col = next((c for c in cols if 'datetime' in c.lower()), cols[0])
        pred_col = next((c for c in cols if 'predicted' in c.lower() and 'demand' in c.lower()), cols[-1])
        actual_col = next((c for c in cols if 'actual' in c.lower() and 'demand' in c.lower()), None)
        
        error = (df[pred_col] - df[actual_col]).abs()
        mean = error.mean()
        peak = df[pred_col].mean()
        deviation = (error**2).mean()**0.5
        mape = (error / df[actual_col]).mean() * 100

        return jsonify({
            'mean': mean,
            'max': peak,
            'deviation': deviation,
            'mape': mape,
            'columns': cols,
            'rows': len(df),
            'preview_rows': len(preview),
            'datetime_col': datetime_col,
            'pred_col': pred_col,
            'actual_col': actual_col,
            'data': [[str(v) if not isinstance(v, (int, float)) else v for v in row] for row in data]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>')
def download_result(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Results not ready'}), 404
    output_path = job['output']
    if not os.path.exists(output_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(output_path, as_attachment=True,
                     download_name=f'prediction_pipeline{job["pipeline"]}.xlsx')

if __name__ == '__main__':
    app.run(debug=True, port=5000)