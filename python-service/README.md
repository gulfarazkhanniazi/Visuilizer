# Room Visualizer — analysis service

Runs the real floor/wall detection server-side in Python: SegFormer semantic segmentation,
monocular depth estimation (Depth Anything V2), SAM 2 object-mask refinement, and an OpenCV/NumPy
geometry pipeline (vanishing points, robust sub-pixel corner line fitting, shading/depth-based
validation) that separates a wall into one independently selectable, independently designable
region per real architectural corner. The Next.js app calls this directly from the browser.

## Setup (one-time)

```bash
cd python-service
python3.11 -m venv venv          # 3.11 recommended — newer Pythons may lack PyTorch wheels
./venv/bin/pip install -r requirements.txt
```

### SAM 2 setup (optional, recommended)

Refines organic-object masks (plants, flowers — see `ORGANIC_COARSE_BLOB_LABELS` in `vision.py`)
with [Meta's SAM 2](https://github.com/facebookresearch/sam2), a trained promptable segmentation
model, in place of the GrabCut/watershed heuristic — leaf-level precision on irregular silhouettes
instead of a blocky approximation. Entirely optional: `vision.py` falls back automatically to the
original GrabCut/watershed refinement if this isn't set up, so skipping this section is fine.

```bash
git clone --depth 1 https://github.com/facebookresearch/sam2.git /tmp/sam2-src
# Not -e (editable) -- that leaves the installed package pointing back at this clone, so deleting
# /tmp/sam2-src afterwards silently breaks the install. A regular install copies the package in.
SAM2_BUILD_CUDA=0 ./venv/bin/pip install /tmp/sam2-src   # CUDA extension optional, skipped here
rm -rf /tmp/sam2-src

mkdir -p sam2_checkpoints
curl -L -o sam2_checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

Runs on CUDA, Apple Silicon (MPS), or CPU automatically (see `_sam2_device` in `vision.py`). The
tiny checkpoint (~150MB) is deliberate — this only ever refines one already-localized object at a
time from a box prompt, not open-ended detection, so a larger checkpoint buys little here.

## Run

```bash
cd python-service
./venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

First startup downloads both models (SegFormer + Depth Anything V2 Small, roughly a few hundred
MB combined, one-time, cached afterwards). Uses CUDA or Apple Silicon (MPS) automatically when
available, CPU otherwise.
The Next.js app expects it at `http://localhost:8000` by default — override with
`NEXT_PUBLIC_ANALYSIS_SERVICE_URL` if you run it elsewhere.

Both this service and `npm run dev` need to be running for the app to work.

## Regression check

`tests/` guards against silently reintroducing a bug that was already found and fixed once —
`tests/baseline.json` is a snapshot of the detection pipeline's own output (wall/floor area,
region counts, object contamination) on every photo in `test-images/`. It is not a ground-truth
accuracy benchmark (this project has no hand-labeled masks); it only proves a code change didn't
unexpectedly move the pipeline's own numbers on photos already known to matter.

Before considering a change to `vision.py` (or anything `lib/geometry.ts`/`lib/perspective.ts`
depend on for mask shape) done:

```bash
./venv/bin/python tests/check_regression.py
```

Exits non-zero if anything regressed — object-contamination increases and large wall/floor area
swings are hard failures; smaller shifts and region-count changes are printed as warnings. If a
change is *meant* to move these numbers (a real, visually-verified accuracy fix), re-baseline:

```bash
./venv/bin/python tests/capture_baseline.py
```

only after confirming the new output by hand (check the debug overlays), the same way every fix
in this project has been verified — this harness catches *unintended* drift, it doesn't replace
looking at the actual masks.
