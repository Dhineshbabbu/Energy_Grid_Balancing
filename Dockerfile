FROM python:3.11-slim

ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 hfuser && chown -R hfuser /app
USER hfuser

EXPOSE 7860 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "server/app.py"]
