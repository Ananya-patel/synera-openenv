FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Environment defaults (all overridden at runtime via HF Space secrets)
ENV PORT=7860
ENV TASK_ID=task1
ENV SEED=42

EXPOSE 7860

# HF Spaces health check
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "app.py"]
