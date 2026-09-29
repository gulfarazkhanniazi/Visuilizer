# Wall & corner scanner

How a room photo becomes separate, selectable walls: the pipeline, the data it
returns, how to run it, and how its accuracy is measured.

## Running it

```
npm run cv:install     # once: Python deps for the CV service
npm run dev            # API :5178, app :5177, CV service :5179
npm test               # 44 tests: unit, synthetic, API, real-photo regression
```

The CV service (`cv-service/server.py`) is optional. If it is not running, a
scan falls back to the junction scanner and says so in `scan.fallbackReason`.
Environment:

| Variable | Default | Meaning |
|---|---|---|
| `CV_SERVICE_URL` | `http://127.0.0.1:5179` | where Express finds the CV service |
| `CV_SERVICE=off` | — | skip the service (junction scanner only) |
| `ONNX_DEPTH=on` | off | in-process relative-depth fallback, no Python (lower accuracy, see below) |
| `DEPTH_MODEL_{FAST,BALANCED,HIGH}` | `da2-metric-indoor-small` | depth model per quality level (CV service env) |
| `CV_DEVICE` | auto | `cuda` / `mps` / `cpu` |
| `SCANNER=legacy` | `auto` | force the junction scanner |
| `SCAN_DEBUG=1` | — | write debug images for every scan to `server/data/debug/` |

## Before: the junction scanner (still the fallback)

`server/src/autodetect.js` + `corners.js`, unchanged in behaviour:

1. SegFormer-B0 (ADE20K) → wall / floor / ceiling / object class maps, snapped to image edges by a guided filter (`refine.js`).
2. Camera assumed **level**, hfov assumed **70°**; camera height from the floor/ceiling junction ratio and a 2.7 m room-height prior.
3. Wall/floor (and wall/ceiling) **junction pixels** un-projected onto the floor plane.
4. Junction run split into straight pieces left to right (`fitRange`, split-and-merge); each piece is a vertical wall plane.
5. Sobel vertical-edge columns (`verticalCorners`) may split a badly fitting run.
6. Wall pixels raycast against the planes (nearest hit; in practice with no extent test), column bands as backstop.

Where it failed on the test photos (`test/reports/baseline-legacy`): walls were
only as good as the visible skirting. A sofa hid the evidence, a window frame's
vertical edge split a wall, a glass partition became four walls, and a chimney
breast merged into the back wall.

**Inputs:** image path, `{ roomHeight, hfov, includeCeiling }`.
**Output (unchanged contract):** `{ objectList, camera, detected, diagnostics }`.
Each `objectList` entry is `{ name, label, product_surface, order, quad[4], realSize{w,h}, mask{feather, polygons[{mode:'add'|'subtract', points}]}, auto, defaults }`.
**Frontend expectations:** the Renderer builds a homography from `quad` +
`realSize`. Click-to-select (`Visualizer.surfaceAt`) tests `mask.polygons`
back to front. The Studio edits the same fields, and its toast reads
`camera.height`. Names must be unique.

## After: the 3D scanner (`server/src/scanner/`)

```
SegFormer masks ─┐            ┌─ LSD line segments ─> vanishing points ─> focal length
                 │            │
photo ──> CV service: metric depth (Depth-Anything-V2 Metric-Indoor)
                 │
                 ▼
  geometry grid (256–512 cols): classes, depth, depth discontinuities
                 │
  back-projection ─> point cloud ─> normals
                 │
  floor plane (RANSAC) ─> gravity, camera height, pitch/roll ─> world frame
  metric scale: floor+ceiling planes vs room-height prior
                 │
  wall planes: normal-direction peaks ─> offset peaks ─> MSAC ─> membership
               ─> components ─> coplanar merge (only across furniture, only if
               one plane fits both) ─> column-profile split (steps/bends)
               ─> validation (residuals, normals, inliers, semantics)
                 │
  corners: plane ∩ plane (or step between parallel planes), projected,
           scored on 8 kinds of evidence; walls whose corner is rejected and
           that one plane explains are merged
                 │
  extents: snapped to validated corners, extended behind occluders and to
           the frame edge ─> quads, polygon3D, wall∩floor/ceiling boundaries
                 │
  raycast every pixel against the final planes ─> exclusive per-wall masks,
           occluded masks, unassigned pixels ─> polygons (validated)
```

| File | Stage |
|---|---|
| `depth.js` | CV service client; opt-in ONNX fallback; SAM client |
| `pointcloud.js` | grid, depth edges, back-projection, normals, RANSAC floor/ceiling, world frame |
| `vanishing.js` | vanishing points → focal length |
| `planes.js` | wall plane detection, merge/split rules, validation |
| `walls.js` | corners and evidence, extents, quads, boundaries |
| `assign.js` | per-pixel raycast and ownership |
| `sam.js` | SAM 2 silhouette refinement (quality `high`) |
| `run.js` | orchestration, public shapes, debug images |
| `config.js` | every threshold and weight |

### Decisions worth knowing

- **Walls are vertical planes.** This is a physical prior, not a Manhattan one. Corner angles are measured, and 45° or 135° corners come out as measured.
- **Focal length.** It comes from, in order: an explicit caller value, EXIF (`FocalLengthIn35mmFilm`, now kept at upload), vanishing points, then 70°. Self-calibrating from depth (making walls perpendicular to the floor) was tried and rejected: monocular depth compression biases it to ~105° on photos Depth Pro puts at 55–70°.
- **Scale.** Monocular "metric" depth put ceilings at 4–7 m in ordinary rooms. When floor and ceiling planes are both visible, their ratio plus the 2.7 m prior fixes the scale (the prior the old scanner used, on better geometry). Otherwise metric depth is trusted if the camera height is plausible, else a camera-height prior applies.
- **An RGB edge can never make a corner.** A corner must be a plane intersection or a step between parallel planes. It must score ≥ 0.5 on the weighted evidence, and an observed corner also needs depth or normal support ≥ 0.35. The old Sobel columns are still evaluated, and when the same plane continues on both sides they are rejected with the reason recorded.
- **Hidden corners.** Where something nearer stands in front of the corner line, the image evidence is excluded (not counted against it). An `occlusion` score checks that the corner really is covered, and the corner is reported with `visible: false, inferred: true` and confidence × 0.85.

### Corner pixel ownership (deterministic)

1. Only planes whose ray hit lies inside their wall's extent compete. Extents are snapped to validated corners, so past a corner the other wall is out of the running.
2. The lowest depth disagreement wins, with a small bonus for the plane the fit assigned the pixel to.
3. Agreements within 0.4% are a tie, and the nearer hit wins. At a corner line both planes are hit at the same depth, so this reduces to "which side of the projected intersection line is the pixel centre on".
4. An exact tie goes to the lower wall index.

Each pixel is written to at most one wall. A wall pixel no plane explains is
left unassigned, not forced onto a wall.

### Output

`POST /api/rooms/:id/auto-detect` accepts `{ quality?: 'fast'|'balanced'|'high', hfov?, roomHeight?, includeCeiling?, debug? }`.
It returns the old `{ room, camera, detected }` plus `scan`:

```js
scan = {
  version: 'scan-1', scanner: 'geometry3d' | 'legacy', quality, fallbackReason?,
  walls: [{
    id: 'wall_01', objectName: 'left_wall', selectable,
    plane: {a,b,c,d} /* world: X right, Y up, Z forward; floor Y=0 */, planeCamera, normal,
    polygon2D, polygon3D, quad, realSize, corners: ['corner_01'],
    visibleMask: {polygons}, occludedMask: {polygons},
    confidence, confidenceLabel, hiddenLength, height: {value, source},
    boundaries: { floor: {observedFraction, medianErrorPx}, ceiling },
    validation: { medianRelResidual, p95RelResidual, inlierRatio, normalConsistency, verticalityDeg, segmentationConsistency },
  }],
  corners: [{
    id: 'corner_01', wallA, wallB, type: 'intersection' | 'step',
    position2D, position3D, segment2D, segment3D, angle, concave,
    visible, inferred, confidence, confidenceLabel,
    evidence: { planeIntersection, depth, normals, lines, segmentation, rgb, roomGeometry, occlusion },
    reason: { ...per-evidence verdicts },
  }],
  rejectedCorners: [{ source, wallA, wallB, reason, evidence, position2D }],
  camera: { focalPx, hfov, focalSource, height, pitchDeg, rollDeg, scaleSource, up, ceilingHeight },
  confidenceBands, metadata: { timings, modelVersions: { segmentationModel, depthModel, depthLicense, geometryVersion, cvService }, debugDir },
}
```

Each wall's `objectList` entry also gains `scanId`, `confidence` and
`geometry`, and the Studio shows the confidence next to each wall. A compact
`scan` summary is saved in the room's `settings.scan`. Every field the
frontend already read keeps its meaning.

### Quality levels

| | grid | lines/VP | SAM 2 | typical geometry time |
|---|---|---|---|---|
| `fast` | 256 | — | — | ~150 ms |
| `balanced` (default) | 384 | ✓ | — | ~250 ms |
| `high` | 512 | ✓ | ✓ (furniture/wall boundaries) | +0.5–2.8 s |

Plus segmentation (0.7–5.6 s, unchanged) and depth (0.1–0.3 s on Apple MPS; cached per image).

### Debug mode

`debug: true` (development) or `SCAN_DEBUG=1` writes to `server/data/debug/<photo>-<time>/`:
`01-original` `02-segmentation` `03-depth` `04-depth-boundaries` `05-normal-map`
`06-point-cloud` (top-down) `07-plane-detection` `08-line-detection` (+ vanishing points)
`09-corner-candidates` (✓/✗ with scores) `10-validated-corners` (evidence per corner)
`11-wall-polygons` (quads) `12-final-wall-mask` (occluded hatched) `13-raycast-result`
(unassigned in white) `14-final-room-model` (top-down with corners), plus `scan.json`.

## Models and licences

| Model | Use | Licence |
|---|---|---|
| SegFormer-B0 ADE20K | segmentation (unchanged) | as before |
| Depth-Anything-V2 **Metric-Indoor-Small** | default depth | **Apache-2.0** |
| Depth-Anything-V2 Metric-Indoor Base/Large | opt-in | CC-BY-NC-4.0 (non-commercial) |
| Apple Depth Pro | opt-in | apple-amlr (research only) |
| SAM 2.1 hiera-small | `high` refinement | Apache-2.0 |

Measured on the six test photos with GT-free geometry checks (`scripts/eval-depth.js`):

| | time | floor residual | wall residual | wall tilt |
|---|---|---|---|---|
| DA-V2 Small | 0.6 s | 0.28 % | 0.44 % | 6.6° |
| DA-V2 Base | 0.8 s | 0.25 % | 0.32 % | 4.4° |
| Depth Pro | 7.8 s | 0.16 % | 0.30 % | 3.6° |

The ONNX fallback (`ONNX_DEPTH=on`) aligns *relative* depth to the floor under
a level-camera assumption. On the test photos it produced clearly worse walls
(one room came out as six), so it is opt-in.

## Validation

- **Synthetic** (`test/synthetic/`): a ray tracer renders rooms with exact ground truth, and depth is corrupted like a monocular network's output (±2 % smooth bias, noise, blur, wrong scale). There are 17 scenes: 2 and 3 walls, diagonal (pitched + yawed), sofa, wardrobe-hidden corner, window, door frame, painting, hard shadow, weak contrast, same colour, 135° corner, bright window, dark room, chimney breast, pitched camera, 40 % wrong depth scale. Current results (`npm run test:synthetic -w server`): **every wall found, 0 merges, 0 splits, 0 false and 0 missed corners**; mean IoU ≥ 0.989; boundary F1 ≥ 0.995; corner error 0.1–1.7 px; corner angle error ≤ 7.5° (the pitched/yawed scene); hidden corner reported as inferred. The larger angle errors come from the focal length when no vanishing point is found; with the true focal they are ≤ 0.4°.
- **Real photos** (`test/regression.test.js`, expectations in `test/fixtures/real-expectations.json`, verified by eye): wall counts, names, corner counts, selection through the frontend's own `pointInMask` (≥ 97 % of clicks select the wall, ≤ 1 % overlap), and stable IDs across rescans.
- **API** (`test/api.test.js`): the real server on a throwaway DB, with upload → auto-detect both with and without the CV service.

### Known limitations

- **Focal length is the main uncertainty.** When no EXIF and no reliable vanishing points exist, 70° is assumed. Estimates on the test photos differed from Depth Pro's by 3–17°. A wrong focal skews corner angles and tile perspective, not wall separation.
- **Small wall pieces** (a panel's side return, a window reveal) are real planes and are reported. The thinnest are marked `selectable: false`.
- **Very small or fragmented wall regions** (a far wall seen between bar stools) are left unassigned rather than guessed.
- The real-photo set is 6 images. The spec asks for 20–50: `node scripts/scan-report.js --images <dir> --debug` runs any folder and writes overlays, JSON and debug images for review.
