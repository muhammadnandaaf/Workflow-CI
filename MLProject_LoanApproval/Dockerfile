FROM python:3.10.16-slim 

# Set working directory
WORKDIR /app

# Salin semua file dari MLProject_LoanApproval ke /app di dalam image
COPY . /app/

RUN pip install --no-cache-dir mlflow>=2.22.0 pandas scikit-learn joblib numpy matplotlib seaborn cloudpickle==3.1.1 psutil==7.0.0 scipy==1.15.2