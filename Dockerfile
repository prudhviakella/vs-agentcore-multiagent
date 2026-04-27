FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

# AGENT_NAME passed via --build-arg (e.g. "supervisor", "research")
ARG AGENT_NAME
ENV AGENT_NAME=${AGENT_NAME}
ENV SSM_PREFIX=/vs-agentcore-multiagent/prod
ENV AGENT_ENV=prod

COPY agents/${AGENT_NAME}/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/${AGENT_NAME}/ ./agents/${AGENT_NAME}/
COPY core/ ./core/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD python -m agents.${AGENT_NAME}.app
