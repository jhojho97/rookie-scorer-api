---
title: Rookie Scorer API
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Rookie Research-Potential Scoring API

FastAPI backend that scores an accounting PhD rookie's future research potential
from an uploaded CV (+ optional job-market paper), reproducing the Ke & Long
(2026) XGBoost C/D/E ensemble. Companion Streamlit frontend calls this over HTTP.

## Endpoints
- `GET /health` — liveness (no auth).
- `POST /predict` — score one candidate. Header `X-API-Key`; multipart body
  `cv` (required), `jmp` (optional). Returns prediction + SHAP `top_factors`.
- `POST /predict/batch` + `GET /jobs/{job_id}` — batch a zip of candidate folders.

## Required Space secrets
`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `API_TOKEN` (shared client secret),
`ALLOWED_ORIGINS` (frontend Space URL). Optional: `ROOKIE_TARGET`
(default `pub_w_top_5pct`).

Interactive docs at `/docs` once running.
