# Homework 05 — Deploying Machine Learning Models

> Official assignment: [ML Zoomcamp 2026 Homework 5](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/05-deployment/homework.md)

## Overview

This homework packages the cohort's frozen lead-scoring model for repeatable
local inference, an HTTP API, and a Docker container. The checked-in model is
loaded with its locked Python dependencies; it is not retrained as part of the
inference workflow.

## Results

| Question | Topic | Documented result | Evidence status |
| --- | --- | --- | --- |
| Q1 | Local `uv` version | `uv 0.13.5` | Verified execution (environment check; not graded) |
| Q2 | Locked Scikit-Learn version | `1.7.2` | Verified from the local locked environment |
| Q3 | Direct model inference | `0.533` | Verified with `q3.py` |
| Q4 | FastAPI inference | `0.770` | Verified locally (`0.769799` before rounding) |
| Q5 | Docker Python base image | `python:3.11.15-slim-bookworm` | Verified from `Dockerfile` |
| Q6 | Container inference | `0.770` | Verified in Docker (`0.769799` before rounding) |

## Project structure

```text
homework_05/
├── pipeline.bin              # Frozen model artifact
├── feature_defaults.json     # Training-time numeric imputation values
├── model_metadata.json       # Artifact and dataset provenance checksums
├── model.py                  # Input normalization shared by scripts and API
├── predict.py                # FastAPI application (`/health`, `/predict`)
├── q3.py / q6_test.py        # Question-specific inference clients
├── smoke_test.py             # Deterministic local-inference smoke check
├── Dockerfile                # Canonical container definition
├── pyproject.toml / uv.lock  # Standalone locked deployment environment
├── homework.md               # Course assignment material
└── notes.txt                 # Original commands, evidence, and answers
```

`train.py` records how the frozen artifact was produced. It requires the
course training CSV and is not needed to reproduce the submitted inference
results.

## Requirements

- Python 3.11.15
- [uv](https://docs.astral.sh/uv/)
- Docker, for the container verification

Run all commands below from the repository root unless a command changes into
this directory.

## Setup with `uv`

```bash
cd src/mlzoomcamp/homework_05
uv sync --locked
uv run python --version
uv run python -c "import sklearn; print(sklearn.__version__)"
```

The deployment environment is self-contained in this directory. Use the
checked-in `uv.lock`; do not regenerate it for the graded workflow.

## Local verification and inference

Verify the frozen artifacts and deterministic inference:

```bash
cd src/mlzoomcamp/homework_05
sha256sum pipeline.bin feature_defaults.json
uv run python smoke_test.py
uv run python q3.py
```

The expected SHA-256 for `pipeline.bin` is:

```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1
```

`model_metadata.json` records the expected hashes for both the model and its
feature defaults.

## FastAPI service

Start the local API:

```bash
cd src/mlzoomcamp/homework_05
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

In another terminal, check service health and send the Q4 lead:

```bash
curl -s http://localhost:9696/health
uv run python q6_test.py http://localhost:9696/predict
```

`GET /health` reports service status and the SHA-256 values of the loaded
model and feature-default artifacts. `POST /predict` returns a numeric
`conversion_probability` and a boolean `conversion` flag.

## Docker

Build the canonical container from the homework directory:

```bash
cd src/mlzoomcamp/homework_05
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

With the container running, verify Q6 from another terminal:

```bash
cd src/mlzoomcamp/homework_05
curl -s http://localhost:9696/health
uv run python q6_test.py http://localhost:9696/predict
```

The Dockerfile pins Python `3.11.15-slim-bookworm` and the `uv` image by
digest, installs the checked-in lockfile, and copies the frozen serving
artifacts and smoke client into the image.

## Notes

- The six answers, original terminal evidence, and closing audit are retained
  in [`notes.txt`](notes.txt).
- The official release files and frozen model artifact are intentionally kept
  under version control for reproducibility.
- Local virtual environments and Python caches are ignored and are not part of
  the published homework.
