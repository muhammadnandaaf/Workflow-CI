FROM continuumio/miniconda3:latest 

# Set working directory
WORKDIR /app

COPY conda.yaml .

RUN conda env create -f conda.yaml

COPY . .

# RUN pip install --no-cache-dir mlflow>=2.22.0 pandas scikit-learn joblib numpy matplotlib seaborn cloudpickle==3.1.1 psutil==7.0.0 scipy==1.15.2