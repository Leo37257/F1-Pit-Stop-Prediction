FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/hf/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/champion/ models/champion/
COPY dashboard/ dashboard/
ENV MLFLOW_TRACKING_URI=file:///tmp/mlruns
EXPOSE 7860
# Hugging Face Spaces expects 7860; Render (and most PaaS) pass the port in $PORT.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
