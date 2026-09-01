#!/bin/bash
# Runs the Python backend with two layers of restart:
#   1. uvicorn --reload restarts the app whenever vision.py/main.py changes.
#   2. This loop restarts uvicorn itself if the process ever dies/crashes/gets killed.
# Ctrl+C stops it for good.
cd "$(dirname "$0")"
source venv/bin/activate
while true; do
  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
  echo "[run-dev] uvicorn exited ($(date)) - restarting in 2s..."
  sleep 2
done
