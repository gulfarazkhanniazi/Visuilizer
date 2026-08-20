# Room Visualizer — analysis service

Runs the real floor/wall detection (SegFormer semantic segmentation + OpenCV corner detection)
server-side in Python. The Next.js app calls this directly from the browser.

## Setup (one-time)

```bash
cd python-service
python3.11 -m venv venv          # 3.11 recommended — newer Pythons may lack PyTorch wheels
./venv/bin/pip install -r requirements.txt
```

## Run

```bash
cd python-service
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

First startup downloads the model (~a few hundred MB, one-time, cached afterwards).
The Next.js app expects it at `http://localhost:8000` by default — override with
`NEXT_PUBLIC_ANALYSIS_SERVICE_URL` if you run it elsewhere.

Both this service and `npm run dev` need to be running for the app to work.
