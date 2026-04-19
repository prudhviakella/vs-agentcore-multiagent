FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy agent + shared layers
COPY agents/${AGENT_NAME%-agent}/ ./agents/${AGENT_NAME%-agent}/
COPY core/ ./core/
COPY shared/ ./shared/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "agents.${AGENT_NAME%-agent}.app"]
