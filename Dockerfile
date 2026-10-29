FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Flask port
EXPOSE 5000

# Run with gunicorn
CMD ["sh", "-c", "python -m src.models.train_models && python -m src.evaluation.pre_deploy_gate && gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 src.api.app:app"]
