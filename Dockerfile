# Rookie Research-Potential scoring API — FastAPI backend for a HF Docker Space.
# Python 3.13 matches the validated env (AUC 97.3) that pickled shap_models/*.pkl.
FROM python:3.13-slim

# gcc kept as insurance for any package without a manylinux wheel on 3.13.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + models + SHAP background CSV. ROOT resolves to /app at runtime,
# and inference.py os.chdir(ROOT), so bundled data files line up.
COPY . .

# Bind to $PORT when the host injects one (Render sets PORT, default 10000);
# fall back to 7860 for local runs / HF Spaces. Shell form so ${PORT} expands.
EXPOSE 7860
CMD uvicorn deepseek_pipeline.webapp.app:app --host 0.0.0.0 --port ${PORT:-7860}
