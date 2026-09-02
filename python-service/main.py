from fastapi import FastAPI, File, Form, HTTPException, UploadFile # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]

from vision import analyze_image, get_depth_estimator, get_metric_depth_estimator, get_segmenter

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
    # Metric depth is what gives tile scale a real unit. Warmed here too, and deliberately
    # tolerant: if the checkpoint can't be fetched, analysis still works and tile scale falls back
    # to an assumed room rather than the whole service failing to start.
    try:
        get_metric_depth_estimator()
    except Exception as exc:  # noqa: BLE001
        print(f"[warm] metric depth unavailable, tile scale will use assumed room size: {exc}")


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

    # Every upload used to be written to /tmp/room_visualizer_uploads here, unconditionally —
    # regardless of the `debug` flag, so a normal request kept a copy of a photo of someone's home
    # on the server's disk with nothing ever deleting it. On Windows the path also resolved to
    # C:\tmp rather than any temp directory. Removed rather than made conditional: `debug=True`
    # already returns the full intermediate bundle in the response, which is what a developer
    # inspecting a bad detection actually needs, so keeping the photo on disk buys nothing.

    try:
        result = analyze_image(image_bytes, target_width, target_height, debug=debug)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return result
