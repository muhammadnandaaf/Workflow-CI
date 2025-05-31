# Monitoring dan Logging/app_serving.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import os
import numpy as np

app = Flask(__name__)

# --- METRIK PROMETHEUS (TARGET MINIMAL 10 UNTUK ADVANCE) ---
PREDICTIONS_TOTAL = Counter('loan_model_predictions_total', 'Total predictions made.')
PREDICTION_ERRORS_TOTAL = Counter('loan_model_prediction_errors_total', 'Total prediction errors.')
PREDICTION_LATENCY_SECONDS = Histogram('loan_model_prediction_latency_seconds', 'Prediction latency in seconds.')
# Tambahkan metrik lain di sini seiring Anda mengembangkannya...

# --- MUAT MODEL DARI DALAM IMAGE DOCKER ---
# Asumsi di Dockerfile Kriteria 3, model disalin ke /app/trained_model.pkl
MODEL_PATH = "/app/trained_model.pkl" 
# Asumsi referensi kolom juga disalin jika diperlukan oleh skrip serving di Docker
# REFERENCE_DATA_PATH = "/app/train_pca.csv" 

model = None
EXPECTED_FEATURE_NAMES = [] # Akan diisi dari file referensi jika ada

try:
    # Coba muat nama kolom fitur jika file referensi ada di image
    # if os.path.exists(REFERENCE_DATA_PATH):
    #     df_train_sample = pd.read_csv(REFERENCE_DATA_PATH, nrows=1, encoding='latin1')
    #     EXPECTED_FEATURE_NAMES = [col for col in df_train_sample.columns if col.lower() != 'loanapproved']
    #     print(f"Nama fitur yang diharapkan (dari image): {EXPECTED_FEATURE_NAMES}")
    # else:
    #     print(f"Peringatan: File referensi fitur {REFERENCE_DATA_PATH} tidak ditemukan di dalam image.")
    #     # Anda HARUS punya cara lain untuk tahu nama & urutan fitur jika ini terjadi
    #     # Misalnya, hardcode jika sudah pasti, atau model Anda menerima input tanpa nama kolom

    # Untuk sekarang, kita hardcode contoh nama fitur PCA jika referensi tidak dimuat
    # GANTI INI DENGAN NAMA KOLOM FITUR ANDA YANG SEBENARNYA SETELAH PCA
    if not EXPECTED_FEATURE_NAMES:
        EXPECTED_FEATURE_NAMES = [f'PC{i+1}' for i in range(10)] # Contoh jika Anda punya 10 PC
        print(f"Menggunakan nama fitur PCA default: {EXPECTED_FEATURE_NAMES}")


    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH} di dalam image.")
    model = joblib.load(MODEL_PATH)
    print(f"Model berhasil dimuat dari dalam image: {MODEL_PATH}")
except Exception as e:
    print(f"KRITIKAL: Gagal memuat model atau referensi fitur dari image. Error: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    global model, EXPECTED_FEATURE_NAMES
    if model is None:
        PREDICTION_ERRORS_TOTAL.inc()
        return jsonify({'error': 'Model tidak tersedia.'}), 500

    try:
        data_json = request.get_json(force=True)
        if not isinstance(data_json, dict):
            PREDICTION_ERRORS_TOTAL.inc()
            return jsonify({'error': 'Input harus JSON object.'}), 400

        # Asumsi input JSON adalah dictionary flat: {"PC1": val1, "PC2": val2, ...}
        # atau {"input_features": [val1, val2, ...]}

        input_values = []
        if "input_features" in data_json and isinstance(data_json["input_features"], list):
            if len(data_json["input_features"]) != len(EXPECTED_FEATURE_NAMES):
                PREDICTION_ERRORS_TOTAL.inc()
                return jsonify({'error': f'Jumlah fitur di "input_features" ({len(data_json["input_features"])}) tidak sesuai. Diharapkan {len(EXPECTED_FEATURE_NAMES)}.'}), 400
            input_values = data_json["input_features"]
        else: # Asumsi input adalah dictionary flat
            missing_features = [f for f in EXPECTED_FEATURE_NAMES if f not in data_json]
            if missing_features:
                PREDICTION_ERRORS_TOTAL.inc()
                return jsonify({'error': f'Fitur berikut tidak ada: {", ".join(missing_features)}.'}), 400
            input_values = [data_json[f] for f in EXPECTED_FEATURE_NAMES] # Ambil nilai sesuai urutan

        input_df = pd.DataFrame([input_values], columns=EXPECTED_FEATURE_NAMES)

        start_time = time.time()
        prediction_result = model.predict(input_df)
        latency = time.time() - start_time

        PREDICTION_LATENCY_SECONDS.observe(latency)
        PREDICTIONS_TOTAL.inc()

        prediction_value = prediction_result[0]
        if isinstance(prediction_value, np.generic):
            prediction_value = prediction_value.item()

        return jsonify({'prediction': prediction_value, 'latency_seconds': latency})

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        import traceback
        print(f"Error predict: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Kesalahan prediksi: {str(e)}'}), 400

if __name__ == '__main__':
    try:
        start_http_server(8000) # Server Prometheus metrics
        print("Server Prometheus metrics berjalan di http://localhost:8000/metrics")
    except OSError as e:
        print(f"Gagal memulai server Prometheus metrics (port 8000): {e}. Pastikan port tidak digunakan.")

    print("Menjalankan aplikasi Flask serving model di http://localhost:5001 ...")
    # Untuk Docker, Flask dev server cukup. Untuk produksi nyata, Gunicorn lebih baik.
    app.run(host='0.0.0.0', port=5001, debug=False)