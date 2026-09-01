from fastapi import FastAPI, File, Form, HTTPException, UploadFile # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]

from vision import analyze_image, get_depth_estimator, get_segmenter

app = FastAPI(title="Room Visualizer Analysis Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_model():
    # Loads (and downloads, first run) both models at startup instead of on the first request —
    # otherwise whichever user's photo happens to be the first `/analyze` call pays that ~10s
    # load cost live.
    get_segmenter()
    get_depth_estimator()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_width: int = Form(...),
    target_height: int = Form(...),
    debug: bool = Form(False),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    import os
    import time

    debug_dir = "/tmp/room_visualizer_uploads"
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, f"{int(time.time() * 1000)}.jpg"), "wb") as f:
        f.write(image_bytes)

    try:
        result = analyze_image(image_bytes, target_width, target_height, debug=debug)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return result
