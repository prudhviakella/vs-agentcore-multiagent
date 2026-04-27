FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY agents/research/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
COPY agents/research/ ./agents/research/
COPY agents/__init__.py ./agents/__init__.py
COPY core/ ./core/

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["python", "-m", "agents.research.app"]
