# Parlay engine + web UI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/data/sports_data.db

WORKDIR /app

# Dependencies first so code edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY run_pipeline.py ./
COPY data/mock ./data/mock

RUN pip install --no-cache-dir -e '.[web]' \
    && mkdir -p /data

# SQLite lives on a mounted volume; without one the container keeps its own copy
# and every restart starts from an empty database.
VOLUME ["/data"]
EXPOSE 8000

# The app refuses a non-loopback bind unless WEB_ACCESS_TOKEN is set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "src.web.app", "--host", "0.0.0.0", "--port", "8000", "--mock"]
