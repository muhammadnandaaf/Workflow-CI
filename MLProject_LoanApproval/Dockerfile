FROM continuumio/miniconda3:latest 

# Set working directory
WORKDIR /app

COPY conda.yaml .

RUN conda env create -f conda.yaml

COPY . .

COPY model.pkl /app/model.pkl 
COPY app_serving_for_docker.py /app/app_serving_for_docker.py
COPY preprocessing_dataset/train_pca.csv /app/train_pca.csv

RUN pip install --no-cache-dir Flask joblib pandas prometheus_client scikit-learn numpy gunicorn

EXPOSE 5001
EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app_serving:app"]