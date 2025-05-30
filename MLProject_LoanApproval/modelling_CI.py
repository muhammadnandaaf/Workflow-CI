import os
import numpy as np
import warnings
import sys
# import dagshub

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV # Contoh tuning (bisa juga GridSearchCV)
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

# # --- Konfigurasi DagsHub ---
# # Ganti dengan informasi Anda
# DAGSHUB_REPO_NAME = 'model_buildingExp' # Sesuaikan dengan nama repo DagsHub Anda
# DAGSHUB_USERNAME = 'muhammadnandaaf' # Sesuaikan dengan username DagsHub Anda

# # Inisialisasi DagsHub, ini akan otomatis mengkonfigurasi MLflow jika mlflow=True
# print("Menginisialisasi DagsHub...")
# dagshub.init(DAGSHUB_REPO_NAME, DAGSHUB_USERNAME, mlflow=True)
# print(f"DagsHub diinisialisasi untuk repo: {DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}")
# print(f"MLflow Tracking URI sekarang seharusnya otomatis diatur ke DagsHub.")

# # Set Eksperimen MLflow (nama eksperimen Anda)
# MLFLOW_EXPERIMENT_NAME = "Loan_Approval_RF_Advanced" 
# mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
# print(f"Eksperimen MLflow diset ke: {MLFLOW_EXPERIMENT_NAME}")
# # --- Akhir Konfigurasi DagsHub ---

# --- Set Eksperimen MLflow untuk Pelacakan Lokal (di Runner CI) ---
MLFLOW_EXPERIMENT_NAME_LOCAL = "CI_Loan_Approval_RF" 
try:
    experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME_LOCAL)
    print(f"Eksperimen MLflow '{MLFLOW_EXPERIMENT_NAME_LOCAL}' dibuat dengan ID: {experiment_id}")
except mlflow.exceptions.MlflowException as e:
    if "already exists" in str(e).lower(): # Periksa apakah error karena sudah ada
        print(f"Eksperimen MLflow '{MLFLOW_EXPERIMENT_NAME_LOCAL}' sudah ada.")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME_LOCAL)
    else:
        raise # Lemparkan error lain jika bukan karena sudah ada
print(f"Eksperimen MLflow lokal diset ke: {MLFLOW_EXPERIMENT_NAME_LOCAL}")
# --- Akhir Set Eksperimen MLflow Lokal ---

# --- Fungsi Pemuatan Data ---
def load_data(train_path, test_path):
    print("Memuat data...")
    try:
        train_df = pd.read_csv(train_path, encoding='latin1')
        test_df = pd.read_csv(test_path, encoding='latin1')
        print("Data berhasil dimuat.")
        return train_df, test_df
    except FileNotFoundError:
        print(f"Error: Pastikan file ada di {train_path} dan {test_path}")
        return None, None

# --- Fungsi Utama ---
def train_random_forest():
    # Path relatif terhadap lokasi MLproject (MLProject_LoanApproval/)
    train_path = "preprocessing_dataset/train_pca.csv" 
    test_path = "preprocessing_dataset/test_pca.csv"

    train_df, test_df = load_data(train_path, test_path)

    if train_df is None or test_df is None: # Pastikan kedua df tidak None
        print("Gagal memuat data. Skrip dihentikan.")
        return

    print("Memisahkan fitur dan target...")
    X_train = train_df.drop("LoanApproved", axis=1)
    y_train = train_df["LoanApproved"]
    X_test = test_df.drop("LoanApproved", axis=1)
    y_test = test_df["LoanApproved"]

    print("Menyiapkan parameter grid untuk RandomizedSearchCV...")
    param_dist = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=300, num=3)], # Kurangi sedikit untuk CI
        'max_features': ['sqrt', 'log2'], 
        'max_depth': [int(x) for x in np.linspace(10, 30, num=3)] + [None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True] # Mungkin pilih satu saja untuk CI
    }

    print("Memulai MLflow Run...")
    current_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "Tidak_Terdeteksi_Dalam_Script"
    print(f"MLflow Run ID aktif: {current_run_id}")

    mlflow.set_tag("Model_Type", "RandomForest_CI")
    mlflow.set_tag("Run_Context", "GitHub_Actions_CI")
    mlflow.set_tag("mlflow.runName", "CI_RF_Training_Run")

    print("Melakukan RandomizedSearchCV...")
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                        n_iter=10, cv=2, verbose=1, # Kurangi n_iter dan cv untuk CI
                                        random_state=42, n_jobs=-1, scoring='accuracy')
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV selesai.")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"Parameter Terbaik: {best_params}")

    print("Melakukan Manual Logging...")
    mlflow.log_params(best_params)

    predictions = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    print(f"Akurasi Test: {accuracy:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("best_cv_score", random_search.best_score_)

    print("Menambahkan Logging Kustom...")
    # Log Confusion Matrix sebagai Gambar
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = "confusion_matrix.png" # Akan disimpan di root run MLflow
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, "plots")
    print(f"Confusion Matrix disimpan dan di-log.")

    # Log Classification Report sebagai File Teks
    report_text = classification_report(y_test, predictions, target_names=[str(c) for c in best_model.classes_])
    # Simpan ke file lalu log, atau log teks langsung jika MLflow versi baru mendukung
    report_path = "classification_report.txt" 
    with open(report_path, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(report_path)
    print(f"Classification Report disimpan dan di-log.")
    
    # Log Feature Importances Plot
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    feature_importances_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importances_series.head(15), y=feature_importances_series.head(15).index)
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    fi_path = "feature_importances.png"
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path, "plots")
    print(f"Feature Importances disimpan dan di-log.")

    # Log Model Terbaik
    # Tambahkan signature dan input example untuk praktik terbaik
    signature = infer_signature(X_train, best_model.predict(X_train))
    input_example = X_train.head()

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_random_forest_model_ci", # Beri nama berbeda untuk CI
        signature=signature,
        input_example=input_example
    )
    print("Manual Logging selesai.")

    if active_run: # Cek lagi sebelum mencetak
        print(f"MLflow Run {run_id} selesai. Akan ada di folder mlruns di runner.")

# --- Jalankan Fungsi ---
if __name__ == "__main__":
    train_random_forest()