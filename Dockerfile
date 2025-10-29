FROM python:3.10-slim

WORKDIR /code

# Install basic OS dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt /code/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py /code/

# Expose port 8000
EXPOSE 8000

# Healthcheck calls the /health endpoint of the service
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
