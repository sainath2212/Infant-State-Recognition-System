FROM python:3.10-slim

WORKDIR /app

# wget required for health check
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# In the future, this will also copy src/, models/, etc.
COPY . .

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost:5001/api/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001"]
