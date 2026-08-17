# Two stages: the compiler toolchain never reaches the runtime image, and the
# process does not run as root.
#
# Build context is the repository ROOT, not service/, because the service
# imports core/, routing/, ml/ and experiments/. Those are the scientific core
# of the project and deliberately do not live inside the web application.

FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY service/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /install /usr/local

COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser routing/ ./routing/
COPY --chown=appuser:appuser ml/ ./ml/
COPY --chown=appuser:appuser experiments/ ./experiments/
COPY --chown=appuser:appuser service/ ./service/
COPY --chown=appuser:appuser README.md ./README.md

RUN mkdir -p ml/checkpoints ml/results experiments/results && \
    chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# --workers 1 is REQUIRED, not a default. service/state.py holds a module-level
# singleton network source and router set; a second worker would own a divergent
# copy of the network and the dashboard would flicker between two realities.
CMD ["python", "-m", "uvicorn", "service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
