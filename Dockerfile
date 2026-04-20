FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# AGENT_NAME passed via --build-arg (e.g. "research", "supervisor")
ARG AGENT_NAME
ENV AGENT_NAME=${AGENT_NAME}

COPY agents/${AGENT_NAME}/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent module + shared core
COPY agents/${AGENT_NAME}/ ./agents/${AGENT_NAME}/
COPY core/ ./core/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD python -m agents.${AGENT_NAME}.app
