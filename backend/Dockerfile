FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HARMONIZER_RUNTIME_DIR=/app/backend/.runtime \
    HARMONIZER_SOUNDFONT=/usr/share/sounds/sf2/TimGM6mb.sf2

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        fluidsynth \
        libsndfile1 \
        timgm6mb-soundfont \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/backend/requirements.txt

COPY ml /app/ml
ARG HARMONIZER_INSTALL_ML_EXTRAS=0
RUN if [ "$HARMONIZER_INSTALL_ML_EXTRAS" = "1" ] \
        && [ -f /app/ml/requirements.txt ]; then \
        python -m pip install --no-cache-dir -r /app/ml/requirements.txt; \
    fi

COPY contracts /app/contracts
COPY backend /app/backend

RUN useradd --create-home --uid 10001 harmonizer \
    && mkdir -p /app/backend/.runtime /app/backend/timbres \
    && chown -R harmonizer:harmonizer /app/backend/.runtime /app/backend/timbres

USER harmonizer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
