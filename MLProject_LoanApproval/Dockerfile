FROM python:3.10.16-slim 

# Set working directory
WORKDIR /app

# Salin semua file dari MLProject_LoanApproval ke /app di dalam image
COPY . .

RUN pip install --no-cache-dir mlflow>=2.22.0 pandas scikit-learn joblib numpy matplotlib seaborn
