# Monitoring dan Logging/app_serving.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import os
import numpy as np

app = Flask(__name__)

# --- METRIK PROMETHEUS (10 METRIK) ---
# 1. Total Prediksi yang Dibuat
PREDICTIONS_TOTAL = Counter(
    'loan_model_predictions_total',
    'Total number of predictions made.'
)
# 2. Total Error Selama Prediksi
PREDICTION_ERRORS_TOTAL = Counter(
    'loan_model_prediction_errors_total',
    'Total number of errors encountered during loan approval prediction.'
)
# 3. Latensi (Waktu Pemrosesan) Prediksi
PREDICTION_LATENCY_SECONDS = Histogram(
    'loan_model_prediction_latency_seconds',
    'Latency of loan approval predictions in seconds.'
)
# 4. Jumlah Fitur pada Request Terakhir (Gauge)
LAST_INPUT_FEATURES_COUNT = Gauge(
    'loan_model_last_input_features_count',
    'Number of features in the last input request.'
)
# 5. Distribusi Skor Prediksi (Probabilitas Kelas Positif)
PREDICTION_SCORE_DISTRIBUTION = Histogram(
    'loan_model_prediction_score_distribution',
    'Distribution of prediction scores/probabilities (for positive class).'
)
# 6. Jumlah Request ke Endpoint /predict
PREDICT_ENDPOINT_REQUESTS_TOTAL = Counter(
    'loan_model_predict_endpoint_requests_total',
    'Total requests to /predict endpoint.'
)
# 7. Jumlah Request dengan Input Tidak Valid/Struktur Salah
INVALID_INPUT_STRUCTURE_TOTAL = Counter(
    'loan_model_invalid_input_structure_total',
    'Total requests with invalid input JSON structure or missing expected keys.'
)
# 8. Versi Model yang Aktif
ACTIVE_MODEL_VERSION = Gauge(
    'loan_model_active_version',
    'Version of the model currently being served',
    ['model_version_label']
)
# 9. Distribusi Nilai Fitur Input Tertentu (Contoh untuk 'AnnualIncome' dan 'PC1')
INPUT_FEATURE_VALUE_DISTRIBUTION = Histogram(
    'loan_model_input_feature_value_distribution',
    'Distribution of a specific input feature value.',
    ['feature_name']  # Label untuk nama fitur
)
# 10. Jumlah Prediksi per Kelas yang Diprediksi
PREDICTIONS_BY_CLASS_TOTAL = Counter(
    'loan_model_predictions_by_class_total',
    'Total number of predictions for each class.',
    ['predicted_class']  # Label untuk kelas yang diprediksi
)

# --- MULAI SERVER METRIK PROMETHEUS ---
METRICS_PORT = 8000
try:
    start_http_server(METRICS_PORT)
    print(f"--- Server Prometheus metrics BERHASIL dimulai di port {METRICS_PORT} (endpoint: /metrics) ---")
except OSError as e:
    print(f"--- KRITIKAL: Gagal memulai server Prometheus metrics (port {METRICS_PORT}): {e}. ---")
except Exception as e:
    print(f"--- KRITIKAL: Error tidak diketahui saat memulai server Prometheus metrics: {e} ---")

# --- MUAT MODEL DAN DEFINISIKAN NAMA FITUR YANG DIHARAPKAN ---
MODEL_PATH = "/app/model.pkl"  # Sesuai Dockerfile
model = None
MODEL_VERSION_LABEL = "1.0.1"  # Ganti dengan versi model Anda

# Daftar nama fitur berdasarkan input Anda (tanpa 'LoanApproved')
EXPECTED_FEATURE_NAMES = [
    'AnnualIncome', 'EmploymentStatus', 'EducationLevel', 'LoanDuration',
    'MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus',
    'MonthlyDebtPayments', 'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
    'NumberOfCreditInquiries', 'BankruptcyHistory', 'LoanPurpose',
    'PreviousLoanDefaults', 'PaymentHistory', 'LengthOfCreditHistory',
    'TotalLiabilities', 'MonthlyIncome', 'UtilityBillsPaymentHistory',
    'JobTenure', 'NetWorth', 'BaseInterestRate', 'pc1_1', 'pc1_2',
    'pc2_1', 'pc2_2', 'pc2_3', 'pc3_1', 'pc3_2', 'pc3_3', 'pc4_1', 'pc4_2'
]
print(f"Nama fitur yang diharapkan model (hardcoded): {EXPECTED_FEATURE_NAMES}")
print(f"Jumlah fitur yang diharapkan: {len(EXPECTED_FEATURE_NAMES)}")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH} di dalam image.")
    model = joblib.load(MODEL_PATH)
    print(f"Model versi '{MODEL_VERSION_LABEL}' berhasil dimuat dari: {MODEL_PATH}")
    ACTIVE_MODEL_VERSION.labels(model_version_label=MODEL_VERSION_LABEL).set(1)
except Exception as e:
    print(f"KRITIKAL: Gagal memuat model. Error: {e}")
    if 'ACTIVE_MODEL_VERSION' in globals():
        ACTIVE_MODEL_VERSION.labels(model_version_label="unknown_error").set(0)
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    global model, EXPECTED_FEATURE_NAMES
    PREDICT_ENDPOINT_REQUESTS_TOTAL.inc()

    if model is None:
        PREDICTION_ERRORS_TOTAL.inc()
        return jsonify({'error': 'Model tidak tersedia. Periksa log server.'}), 500

    try:
        data_json = request.get_json(force=True)

        if not isinstance(data_json, dict):
            PREDICTION_ERRORS_TOTAL.inc()
            INVALID_INPUT_STRUCTURE_TOTAL.inc()
            return jsonify({'error': 'Input harus berupa JSON object.'}), 400

        # Validasi apakah semua fitur yang diharapkan ada di input JSON
        missing_features = [f for f in EXPECTED_FEATURE_NAMES if f not in data_json]
        if missing_features:
            PREDICTION_ERRORS_TOTAL.inc()
            INVALID_INPUT_STRUCTURE_TOTAL.inc()
            return jsonify({'error': f'Fitur berikut tidak ada dalam input: {", ".join(missing_features)}.'}), 400

        # Ambil hanya fitur yang diharapkan dan dalam urutan yang benar
        try:
            input_data_ordered_dict = {feature: data_json[feature] for feature in EXPECTED_FEATURE_NAMES}
        except KeyError as e:
            PREDICTION_ERRORS_TOTAL.inc()
            INVALID_INPUT_STRUCTURE_TOTAL.inc()
            return jsonify({'error': f'Fitur {str(e)} tidak ditemukan dalam data input JSON (setelah validasi).' }), 400

        input_df = pd.DataFrame([input_data_ordered_dict], columns=EXPECTED_FEATURE_NAMES)
        LAST_INPUT_FEATURES_COUNT.set(len(input_df.columns))

        # Implementasi INPUT_FEATURE_VALUE_DISTRIBUTION untuk beberapa fitur contoh
        # Anda bisa memilih fitur lain atau membuat loop jika perlu
        features_to_monitor_dist = ['AnnualIncome', 'pc1_1', 'LoanDuration'] # Contoh fitur
        for feature_name_to_monitor in features_to_monitor_dist:
            if feature_name_to_monitor in input_df.columns:
                try:
                    value = float(input_df[feature_name_to_monitor].iloc[0])
                    INPUT_FEATURE_VALUE_DISTRIBUTION.labels(feature_name=feature_name_to_monitor).observe(value)
                except (ValueError, TypeError):
                    print(f"Peringatan: Nilai untuk {feature_name_to_monitor} tidak valid untuk histogram: {input_df[feature_name_to_monitor].iloc[0]}")


        start_time = time.time()
        prediction_result = model.predict(input_df)
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_df)
        latency = time.time() - start_time

        PREDICTION_LATENCY_SECONDS.observe(latency)
        PREDICTIONS_TOTAL.inc()

        prediction_value = prediction_result[0]
        if isinstance(prediction_value, np.generic):
            prediction_value = prediction_value.item()

        PREDICTIONS_BY_CLASS_TOTAL.labels(predicted_class=str(prediction_value)).inc()

        output = {'prediction': prediction_value}
        if prediction_proba is not None:
            # Asumsi kelas positif (misal, 'LoanApproved' = 1) adalah indeks 1
            # Jika kelas Anda berbeda (misal 0 dan 1, atau string), sesuaikan
            positive_class_index = 1 # Asumsi
            if prediction_proba.shape[1] > positive_class_index:
                 score = prediction_proba[0][positive_class_index]
                 PREDICTION_SCORE_DISTRIBUTION.observe(score)
                 output['probabilities'] = prediction_proba[0].tolist()
                 output['predicted_score'] = score
            else: # Hanya ada satu probabilitas (misal model regresi yang di-threshold)
                 score = prediction_proba[0][0] 
                 PREDICTION_SCORE_DISTRIBUTION.observe(score)
                 output['probabilities'] = prediction_proba[0].tolist()
                 output['predicted_score'] = score
        
        output['latency_seconds'] = latency
        
        return jsonify(output)

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.inc()
        INVALID_INPUT_STRUCTURE_TOTAL.inc() 
        import traceback
        print(f"Error pada endpoint /predict: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 400

if __name__ == '__main__':
    print("Menjalankan aplikasi Flask (mode development) di http://localhost:5001 ...")
    app.run(host='0.0.0.0', port=5001, debug=True)