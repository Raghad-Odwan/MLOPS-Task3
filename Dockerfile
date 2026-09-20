# Small, production image for the inference API only.
# No notebooks, no dev tools, no test files inside.

FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies (not requirements-dev.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the service needs at runtime
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY mlflow.db ./mlflow.db
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
