# Small, production image for the inference API only.
# No notebooks, no dev tools, no test files inside.

FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (not requirements-dev.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs at runtime.
# Note: mlflow.db and mlruns/ are NOT copied from the host. They are
# created fresh inside the container on first startup (see entrypoint.sh),
# using the exact MLflow version installed in this image. This avoids any
# version mismatch between a locally-created mlflow.db and the container.
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY register_model.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]