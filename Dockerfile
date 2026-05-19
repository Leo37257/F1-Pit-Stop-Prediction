FROM apache/airflow:2.9.3-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
