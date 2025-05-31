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
# 4. Jumlah Fitur pada Request Terakhir (Gauge) 
LAST_INPUT_FEATURES_COUNT = Gauge(
    'loan_model_last_input_features_count',
    'Number of features in the last input request.'
)
# 5. Skor Prediksi (jika klasifikasi biner/multikelas dengan probabilitas - Histogram) 
PREDICTION_SCORE_DISTRIBUTION = Histogram(
    'loan_model_prediction_score_distribution',
    'Distribution of prediction scores/probabilities.'
)
# 6. Jumlah Request ke Endpoint /predict (Counter) 
PREDICT_ENDPOINT_REQUESTS_TOTAL = Counter(
    'loan_model_predict_endpoint_requests_total',
    'Total requests to /predict endpoint.'
)
# 7. Jumlah Fitur Input Tidak Sesuai (Counter) 
INVALID_INPUT_FEATURES_TOTAL = Counter( # Ubah nama agar lebih jelas dari prediction_errors
    'loan_model_invalid_input_features_total',
    'Total requests with invalid input feature structure or count.'
)
# 8. Versi Model yang Aktif (Gauge dengan label) 
ACTIVE_MODEL_VERSION = Gauge(
    'loan_model_active_version',
    'Version of the model currently being served',
    ['model_version_label'] # Tambahkan label 'model_version_label'
)

# --- METRIK TAMBAHAN BARU ---
# 9. Distribusi Nilai Fitur Input Tertentu (Histogram dengan label)
# Misalnya, jika Anda punya fitur 'PC1' setelah PCA
INPUT_FEATURE_VALUE_DISTRIBUTION = Histogram(
    'loan_model_input_feature_value_distribution',
    'Distribution of a specific input feature value.',
    ['feature_name'] # Label untuk nama fitur
)

# 10. Jumlah Prediksi per Kelas (Counter dengan label)
PREDICTIONS_BY_CLASS_TOTAL = Counter(
    'loan_model_predictions_by_class_total',
    'Total number of predictions for each class.',
    ['predicted_class'] # Label untuk kelas yang diprediksi
)

# --- MULAI SERVER METRIK PROMETHEUS (DI LUAR if __name__ == '__main__') ---
METRICS_PORT = 8000
try:
    start_http_server(METRICS_PORT)
    print(f"--- Server Prometheus metrics BERHASIL dimulai di port {METRICS_PORT} (endpoint: /metrics) ---")
except OSError as e:
    print(f"--- KRITIKAL: Gagal memulai server Prometheus metrics (port {METRICS_PORT}): {e}. Pastikan port tidak digunakan oleh Gunicorn atau proses lain di dalam container. ---")
except Exception as e:
    print(f"--- KRITIKAL: Error tidak diketahui saat memulai server Prometheus metrics: {e} ---")
# --- AKHIR MULAI SERVER METRIK PROMETHEUS ---

# --- MUAT MODEL DARI DALAM IMAGE DOCKER ---
# Asumsi di Dockerfile Kriteria 3, model disalin ke /app/model.pkl
MODEL_PATH = "/app/model.pkl" 
REFERENCE_DATA_PATH = "/app/train_pca.csv" 

model = None
EXPECTED_FEATURE_NAMES = [] # Akan diisi dari file referensi jika ada

try:
    # Muat nama kolom fitur dari train_pca.csv di dalam image
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"Peringatan Kritis: File referensi fitur '{REFERENCE_DATA_PATH}' tidak ditemukan di dalam image.")
        # Handle kasus ini: bisa error, bisa pakai default, tergantung kebutuhan
        # Jika ini kritikal, sebaiknya raise Exception agar build Docker gagal atau container tidak start.
        # Untuk sekarang, kita coba lanjutkan dengan EXPECTED_FEATURE_NAMES kosong, 
        # tapi ini akan menyebabkan error di /predict jika tidak diisi.
    else:
        df_train_sample = pd.read_csv(REFERENCE_DATA_PATH, nrows=1, encoding='latin1')
        # Ganti 'LoanApproved' dengan nama kolom target aktual Anda jika berbeda
        # Menggunakan .lower() untuk perbandingan case-insensitive
        EXPECTED_FEATURE_NAMES = [col for col in df_train_sample.columns if col.lower() != 'LoanApproved']
        if not EXPECTED_FEATURE_NAMES:
            print(f"Peringatan: Tidak ada nama fitur yang berhasil diekstrak dari '{REFERENCE_DATA_PATH}' setelah mengeluarkan kolom target.")
        else:
            print(f"Nama fitur yang diharapkan model (dimuat dari image): {EXPECTED_FEATURE_NAMES}")
            print(f"Jumlah fitur yang diharapkan: {len(EXPECTED_FEATURE_NAMES)}")

    if not EXPECTED_FEATURE_NAMES:
        print("KRITIKAL: EXPECTED_FEATURE_NAMES kosong. Menggunakan fallback hardcoded (SESUAIKAN JIKA PERLU).")
        EXPECTED_FEATURE_NAMES = [f'PC{i+1}' for i in range(10)] # Fallback, sesuaikan jumlahnya!
        print(f"Menggunakan nama fitur PCA default (fallback): {EXPECTED_FEATURE_NAMES}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH} di dalam image.")
    model = joblib.load(MODEL_PATH)
    MODEL_VERSION_LABEL = "1.0.0" # Atau versi lain yang relevan
    print(f"Model versi '{MODEL_VERSION_LABEL}' berhasil dimuat dari dalam image: {MODEL_PATH}")
    ACTIVE_MODEL_VERSION.labels(model_version_label=MODEL_VERSION_LABEL).set(1)
    print(f"Model berhasil dimuat dari dalam image: {MODEL_PATH}")
except Exception as e:
    print(f"KRITIKAL: Gagal memuat model atau referensi fitur dari image. Error: {e}")
    ACTIVE_MODEL_VERSION.labels(model_version_label="unknown_error").set(0) # Menandakan model tidak aktif/error
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    global model, EXPECTED_FEATURE_NAMES, MODEL_VERSION_LABEL # Tambahkan MODEL_VERSION_LABEL
    PREDICT_ENDPOINT_REQUESTS_TOTAL.inc()

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
    print("Menjalankan aplikasi Flask serving model di http://localhost:5001 ...")
    # Untuk Docker, Flask dev server cukup. Untuk produksi nyata, Gunicorn lebih baik.
    app.run(host='0.0.0.0', port=5001, debug=False)