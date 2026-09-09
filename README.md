# Surface Visualizer

A wall &amp; floor visualizer in the mould of tilesview.ai: upload a room
photograph and its floor and walls are found automatically, then lay any tile in
your catalogue on them with correct perspective, real-world scale, grout, bond
patterns and the room's own lighting. Flat photos and 360° panoramas both work.

```
npm install
npm run seed     # demo rooms, two 360 rooms, a 16-product starter catalogue
npm run dev      # API on :5178, app on :5177
```

Then open <http://localhost:5177>. The first visit to `/admin` claims the
installation by creating an administrator; after that the panel needs a sign-in.
On a server, set `ADMIN_EMAIL` and `ADMIN_PASSWORD` before the first start and
the claim screen never appears.

For production: `npm run build && npm start` — the API serves the built SPA from
the same origin on :5178.

> If `npm run dev` reports the port is in use, a previous copy is still running.
> `taskkill /F /IM node.exe` on Windows, then try again.

---

## How it actually works

The realism does not come from AI. It comes from knowing where each surface is
in the room, and a shader that does three things with that.

### 0. Finding the surfaces in the first place

Upload a photo and the floor and walls are marked for you; the Studio is for
refining that, not for starting from nothing.

**Where the surfaces are.** A semantic segmentation (SegFormer trained on
ADE20K, ~15 MB, run locally) labels every pixel wall / floor / ceiling / sofa /
curtain. The floor class already excludes the furniture standing on it, so the
class map *is* the coverage mask -- it only has to be traced into the same
editable polygons a human would draw (Moore-neighbour contour trace, then
Ramer-Douglas-Peucker down to a few dozen points, with enclosed background
becoming `subtract` holes).

**How they sit in space.** Segmentation says nothing about geometry, and a mask
alone cannot lay a tile. Interior photographs are shot level -- that is the
convention that keeps verticals vertical -- so the horizon runs through the
principal point, and the floor plane reduces to:

```
u = cx + f·X / Z          Z = f·h / (v − cy)
v = cy + f·h / Z          X = (u − cx)·h / (v − cy)
```

X depends only on `h`, not on `f`: **the lateral scale is set by camera height
alone.** So when the ceiling is visible, the per-column ratio of the floor
junction below the horizon to the ceiling junction above it fixes where the
camera sits within an assumed room height, and the metric scale is solved
rather than guessed. On the test photos it lands on ~1.35 m, against 1.38 m
derived by hand.

Each wall is then found from its own junctions: a junction point un-projects
onto that solved floor plane, giving the wall's base line in world space, and a
vertical plane through a known base line is a complete wall. Two details matter
in real rooms:

- a wall pixel only counts as a floor junction if there is *floor* directly
  beneath it, otherwise the bottom edge of a sofa is mistaken for the skirting
  and drags the wall metres out of position;
- skirting is hidden behind furniture far more often than cornice is, so the
  **wall/ceiling** junction is pooled in as well -- a wall is vertical, so both
  junctions constrain the very same base line.

Two more things real rooms need:

- **Walls come back as one blob.** Segmentation has a single "wall" class, so
  the left, back and right walls of an ordinary room arrive connected -- and
  their junction points lie on three different lines, which is why a single fit
  finds none of them. *Sequential RANSAC* pulls out the dominant line, removes
  its inliers and repeats, recovering each wall as its own plane.
- **The corner between two walls is not vertical in the image**, so splitting
  the region into column bands hands a wedge of one wall to the other — and
  that wedge then gets a homography that does not describe it, so its tiles
  come out at the wrong scale and angle. Each fitted plane is a known finite
  rectangle in world space, so every wall pixel is instead tested the way the
  panorama renderer tests one: cast the ray, intersect every plane, keep the
  nearest hit that lands inside that plane's extent. The corner falls out of
  the geometry, and a wall broken in two by a wardrobe stays one surface
  because the fit is done over the whole wall class at once rather than per
  connected blob. Column bands remain only as a backstop for pixels the
  geometry could not place, so a corner never leaves a bare stripe of photo.
- **The class map is snapped to the photograph before anything else happens.**
  The segmenter runs at 512x512 and is right about *what* is where and wrong by
  several pixels about where it stops; blown up to a 2400px photo that becomes
  a wobbling band tens of pixels wide, and the wall material creeps over the
  ceiling line. No amount of polygon smoothing fixes it, because the
  information is not in the class map -- it is in the photograph, where every
  boundary that matters is a strong intensity edge. A **guided filter** (He,
  Sun & Tang) fits a local linear model of the mask against image intensity, so
  the refined mask is forced to be a linear function of the image nearby and
  its transitions land where the image's do. The classes are then made mutually
  exclusive by one argmax, so neighbouring surfaces meet exactly and a pixel no
  class claims -- a sofa, a curtain, a window -- is simply left out, which is
  what makes furniture silhouettes come out crisp.
- **Walls are separated by fitting the junction line piecewise, not by RANSAC.**
  Sequential RANSAC looks for the globally dominant line and discards the
  points that agree with it, which ignores the one thing always true of walls
  in a photograph: from a single viewpoint they are contiguous in image columns
  and they meet at corners. A line fitted from the leftovers of one wall and
  the start of the next is perfectly plausible to RANSAC and geometric nonsense
  in the room. Read left to right instead, the wall/floor junction is a
  piecewise-linear path across the ground plane and the corners are its
  breakpoints, so the fit is split-and-merge: try every breakpoint, keep the
  one that best explains the run, then merge back neighbours that turn out to
  be collinear. Prefix sums over the junction points make the best-fit line
  through any range O(1) -- the sum of squared perpendicular residuals is
  exactly the smaller eigenvalue of the 2x2 covariance -- so trying every
  breakpoint at every level is affordable.
  - A corner is **not** always a turn. A chimney breast, or a window wall set
    back from the one beside it, steps rather than bends, so the test is
    whether two lines explain the run far better than one -- not whether the
    direction changed.
  - A run narrower than about 1.2 m **of wall** is absorbed into whichever
    neighbour explains it better, measured in metres rather than in pixels: a
    run can occupy plenty of columns and still be a metre of wall seen at a
    glancing angle. Discarding slivers instead would leave their columns
    owned by nothing -- a bare stripe of photograph down the join -- and
    shipping them produces two half-metre "walls" where the room has one.

- **Where the junction is hidden, the corner is still visible.** Behind a sofa
  or a run of units the piecewise fit has nothing to read, and two walls come
  back as one badly-fitting plane. But interior photographs are shot level to
  keep verticals vertical -- the same assumption the camera model rests on --
  so a vertical line in the room projects to a vertical line in the image, and
  a wall corner is therefore a *column*. Finding one is a one-dimensional
  search (`corners.js`).

  The work is not detecting vertical edges, it is telling a corner from a
  window frame or a door architrave. Three things separate them: **continuity**
  (a corner runs the full height of the wall, a frame stops), **contrast** (two
  walls at different angles catch the light differently, so brightness steps
  across a corner and does not across a frame, which has the same wall on both
  sides), and **isolation** (frames come in pairs a window's width apart, so
  candidates are thinned to the strongest in a neighbourhood).

  A corner may only divide a run the junction fit already explains *badly*.
  A run that fits well is one straight wall, and a vertical line crossing it is
  a door frame or the edge of a wardrobe -- splitting there would invent a
  corner the room does not have.

- **Ceilings un-project through their own plane.** A ceiling and a floor are
  mirror images about the horizon, and reading one with the other's equation
  inverts the sign of (v - cy) and substitutes the height below the camera for
  the height above it. Ceiling detection is off by default (`includeCeiling`),
  so this only matters when it is switched on.
- **A plane with nothing visible on it is reported, not dropped.** A wall span
  entirely behind a curtain is geometrically fine and has no pixels to tile, so
  it produces no surface; saying so in the diagnostics is what stops "four
  planes, three walls" reading as a bug.
- **A pixel the segmenter is sure is an object is never part of a surface.**
  The comparison above is a majority vote and can be talked round -- a rug lit
  like the floor beside it, a pale curtain against a pale wall -- and being
  talked round means laying tile over a rug, which is the most obviously wrong
  thing this can do. So the occluder map is punched out of every surface
  afterwards, dilated by a pixel: erring towards not tiling an object is
  invisible, erring the other way is not. Measured over four photographs it
  takes rug contamination of the floor from 0.1% to 0.0% and object
  contamination of the walls from 1.3% to 0.5%, for about three points of
  recall -- the safety band around each object, which nobody can see.
- **A hole no object explains is filled, not cut out.** An enclosed gap in a
  wall is a cut-out when something is actually there -- a painting, a socket --
  and segmenter uncertainty otherwise. Cutting the second kind out leaves an
  untiled island in the middle of a wall for no reason, so a hole only becomes
  a `subtract` polygon when the occluder map accounts for at least a third of
  it.
- **The masks are cleaned before they are traced.** A class map is speckled --
  stray wall inside the sofa, stray sofa inside the wall -- and traced as-is
  every speck becomes its own polygon or a spike on someone else's. An
  open-then-close pass removes them, and the contour is re-simplified until it
  is under ~110 points, because every point is a draggable handle in the Studio
  and a 400-vertex outline is not a better mask but an uneditable one.
- **Walls are named the way someone in the room would name them.** A wall that
  runs across the view *and* spans the optical axis is the Back Wall; anything
  else is Left or Right by where it sits in the frame. "Wall 1 / Wall 2 / Wall
  3" is accurate and useless.
- **Some walls show no junction at all** -- furniture pushed flat against the
  wall, ceiling out of frame. That happens precisely when the shot is square-on
  to the wall, so the fallback assumes exactly that and takes the depth from
  where the floor runs out. If the wall also runs off the top of the frame its
  height is unknowable, so the assumed room height is used rather than reading
  the crop as a five-metre wall.

Lines are fitted with RANSAC because segmentation edges are ragged, and a wall
whose fit implies something absurd (wider than 14 m) is dropped rather than
shipped. Detection reports why it rejected anything, so a failure is
diagnosable instead of silent.

Verified on six varied interior photographs: floor and at least one wall found
on every one.

First run downloads the model and takes ~20 s; after that it is ~5 s per photo.
With no internet the app falls back to the Studio and says so.

**The model is the ceiling on all of this.** The pipeline already extracts ~95%
of what SegFormer-B0 labels, so further work on the masks has little left to
win -- what is missing is missing from the class map. `SEG_MODEL` selects a
bigger one: `Xenova/segformer-b4-finetuned-ade-512-512` scores about 50 mIoU on
ADE20K against B0's 37, and finds the pendant lamps B0 paints straight over. It
is not a free win, which is why it is not the default: it costs about twice the
inference time and a much larger one-off download, and on the same photograph
it under-segments a rug that B0 gets right. If it fails to load the detector
falls back to B0 rather than losing the feature.

### 1. Perspective

**Flat photos — one homography per surface.** A floor or a wall is a *plane*.
The mapping from that plane's own metric coordinates to pixels in the photo is
exactly a 3×3 projective homography — no camera pose, no depth estimation, no 3D
reconstruction needed.

In the Studio you drag four corners onto a rectangle you know the real size of
(wall-to-wall along the skirting, say) and type in its dimensions. That is four
point correspondences, which is exactly enough to solve for `H`
(`web/src/engine/homography.js`). The shader applies `H⁻¹` per pixel to get a
position **in metres** on that plane, and looks the tile pattern up there.

This is why a 600 mm tile is 600 mm: the pattern is generated in metres and
projected, rather than an image being stretched to fit.

**360 panoramas — ray/plane intersection.** Every texel of an equirectangular
panorama *is* a direction. Casting that ray at the room's surface planes and
keeping the nearest hit gives both which surface it is and where on it, in
metres. Occlusion falls out of the geometry, so **a 360 room needs no masks at
all**.

Both paths feed the same tiling core (`web/src/engine/tileCore.glsl.js`), so
bond patterns, joints and relighting can never drift apart between them.

### 2. Masking (flat photos)

Each surface owns a list of polygons — `add` for the surface, `subtract` for
whatever sits in front of it (furniture, a rug, skirting). They are stored as
JSON, a few hundred bytes, and rasterised to an antialiased coverage map at
photo resolution when the room loads (`web/src/engine/masks.js`).

Keeping them as geometry rather than a baked image means a room stays editable
forever, and the same polygons drive click-to-select and the area measurement.

### 3. Relighting: the photo lights the tile

This is the part that sells it. The photo already contains every shadow, bounce
and light fall-off in the room — the trick is transferring them onto a new
material without dragging the old floor's own pattern along.

At load time the renderer builds a **lighting plate**: a two-pass Gaussian over
the photo's luminance, keeping both the blurred and the raw value. The blur
radius is the one dial that matters — big enough to erase the previous tile
grid, small enough to keep the shadow under a chair sharp. It is exposed per
room in the Studio.

The surface shader then does, roughly:

```
shade   = (blurredLuminance / referenceLevel) ^ shadeStrength
detail  = 1 + (rawLuminance - blurredLuminance) / referenceLevel * detailAmount
result  = tileColour * shade * detail + gloss * specular
```

`referenceLevel` is the 60th percentile of luminance inside that surface — a
percentile rather than a mean, because a floor full of dark furniture shadows
would otherwise drag the reference down and wash the tile out.

Compositing happens on sRGB-encoded values, the way a photo editor's multiply
blend does; three's colour management is switched off so the transfer function
is not applied twice.

### Bond patterns

Every pattern is resolved analytically in the fragment shader, so all of them
stay perspective-correct and antialiased for free.

| Pattern | Method |
| --- | --- |
| Grid, brick ½, brick ⅓, vertical, vertical brick | Running bond by `mod`, with a per-course offset |
| Diagonal, diagonal brick | The same, rotated 45° before the lookup |
| Basketweave | Groups of `round(L/W)` tiles forming squares of side `L`, alternating orientation |
| Herringbone | Lattice `a = (W, W)`, `b = (−L, L)` with a horizontal tile at `[0,L)×[0,W)` and a vertical tile at `[L,L+W)×[W−L,W)`. The fundamental domain is L-shaped rather than a parallelogram, so the resolver probes back up to `ceil(L/2W)` steps along `a` and one along `b` — bounds measured over the whole plane for ratios 1:1 → 6:1. |

Products can ship several **random faces**; the shader picks one per tile by
hashing the tile id, which stops a laid floor from visibly repeating.

Once a pixel spans a whole tile there is no joint left to resolve, only its
average — so the shader crossfades to the analytic coverage ratio rather than
letting the far field break into moiré.

### Materials

Not everything is laid like a tile. `material` selects one of five rendering
models, and the fragment shader switches on it — the same switch the product
form and the viewer's control panels read, so a material can never be labelled
one thing and drawn as another.

| Model | Materials | What it does |
| --- | --- | --- |
| `module` | tile, marble, granite, stone, quartz, wood, hardwood, engineered, laminate, vinyl, SPC, WPC, carpet tile, wall panel | Discrete units with joints. Everything above applies. |
| `sheet` | wallpaper, broadloom carpet, epoxy | One continuous pattern repeating over the plane. `uTileSize` becomes the pattern repeat; there is no joint, so nothing reads the grout uniforms. Gradients come from the *unwrapped* coordinate, or the wrap would draw a seam at every repeat. |
| `solid` | paint | A flat colour with a sheen and no texture at all. The colour is a uniform rather than a tinted image, so changing it is instant and exact — and a paint product needs no photograph, only a hex value, from which its catalogue swatch is generated. |
| `piece` | rug | A single bounded rectangle centred on the surface and moved by the offset controls. Outside it the shader returns zero coverage and the photograph is simply left alone. |
| `joint` | grout | Draws **only** the joints, in the new colour, over the floor already in the photograph. Line the size and bond up with the existing tile and you have changed the grout without changing the tile — which is what a grout visualizer is. |

The alpha channel is what makes the last two possible: `shadeSurface` returns
rgb *and* the coverage the material claims, so a material can decline to cover
the pixel it was asked about. The quantity calculator follows the same split —
modules are counted in pieces, sheets and paint against their stated coverage,
a rug is one object however big the floor is, and grout is not a field material
at all.

### Lens distortion

A homography models a pinhole camera exactly and a real one approximately, so
on a wide-angle phone photo the tile courses bow away from straight near the
frame edge. The correction is the **division model**, `r_u = r_d / (1 + k1·r_d²)`,
chosen over Brown's polynomial for one reason: the shader has an observed pixel
and needs the ideal one, and that is the direction the polynomial cannot go in
closed form. One coefficient per room, set on a slider in the Studio against a
grid that bends with it, and the area measurement folds in the radial map's own
Jacobian so a corrected photo still measures true.

---

## Features

**Viewer** — every surface is independent: each wall is its own plane and takes
its own product, size, bond and grout (a "link surfaces of the same type"
toggle applies one change to all the walls at once when you want it). Surface
picker, product catalogue with search / category / sort / price, tile size, 9 bond patterns, free rotation and offset, grout width and
colour, bevel, finish (matt → polished), tint, "keep original lighting" and
"surface detail" blending, random face and random rotation, zoom & pan,
click-to-select a surface, undo/redo, fullscreen. The control panels follow the
material: paint gets a colour card and a sheen, wallpaper a pattern repeat, a
rug a position — and no panel of dead controls for the ones that mean nothing.

**Surface markers** — detection is invisible, and a room that opens looking
exactly like the photograph says nothing about what can be tapped. A pin on
each surface shows what was found, what is on it now, and what is still empty.
They introduce the room for a few seconds and then get out of the way, with a
toolbar button to bring them back. Each pin sits at the *pole of
inaccessibility* of its mask rather than the centroid, so it never lands on an
edge, in a cut-out, or on the neighbouring wall's pin.

**Compare** — A/B split slider with swap and "keep this one".

**360°** — look around by dragging, field-of-view zoom, auto-spin, and
gyroscope on phones (asks permission from the user gesture, as iOS requires).

**3D rooms** — upload a glTF/GLB model and tag which of its meshes a customer
may re-tile; everything left untagged keeps the material it shipped with, which
is what the furniture should do. Orbit, dolly and reset. The tiling reads world
metres through a box projection rather than the model's own UVs, deliberately:
authored UVs carry no real-world scale, so a 600 mm tile would be 600 mm only
by luck.

**Quantity calculator** — per-area rows in m² or sq ft, a wastage percentage,
and output in tiles, whole boxes and cost. The **Measure** button fills a row
from the surface's *actual* area: a projective map has Jacobian determinant
`det(H) / (g·x + h·y + i)³`, so summing that over the masked pixels converts
image area straight into metric area — correctly weighting distant pixels,
which cover far more floor than near ones. (Verified against a known rectangle:
7.8400 m² expected, 7.8401 measured.) The reference product makes you type this
number in by hand.

**Save, share, enquire** — save a scheme to come back to, short share links with
a QR code, a wishlist, and an enquiry form that sends the visitor's details
together with the products they had applied.

**Export** — HD (native resolution) and SD (1280 px) images with an optional
watermark, a branded PDF room sheet, and a branded PDF product catalogue.

**Studio** — refine what was detected, or author from scratch: **Auto-detect**
re-runs detection, or add surfaces by hand, drag the four perspective corners
against a half-metre grid, trace masks and cut out occluders, tune the
lighting-plate blur and the lens correction, and preview with a test tile.
Every surface is outlined and labelled at once, not just the one being edited —
overlaps and gaps between neighbouring walls are exactly what is most likely to
be wrong after automatic detection, and invisible if you can only see one at a
time.

**Languages** — the visitor-facing app ships in 24 languages, Arabic included
with the layout mirrored. English is bundled; every other language is a dynamic
import, so a French visitor downloads French and nothing else, and a missing
key falls back to English rather than showing the key. The Studio and Admin
stay in English: they are operated by the business rather than its customers.

**Showroom kiosk** — `?kiosk=1` puts a screen on the shop floor into kiosk
mode: no route through to the Studio or Admin, larger touch targets, an attract
screen, and a reset that clears the last customer's work after a configurable
period of no input.

**Admin** — products (with material, paint colour, price, price unit, pieces
per box, coverage, description, multiple faces), bulk upload, categories,
rooms, 3D rooms, showrooms, branding (name, logo, accent colour, currency,
watermark, contact details), per-feature switches, enquiry inbox, an iframe
embed snippet, and a QR generator for direct links into any single room.

**Accounts** — the admin panel is behind a sign-in: scrypt password hashes,
opaque session tokens in the database, per-account sessions that a password
change revokes, and a login throttle. A fresh install with no account lets the
first person in to create one and closes the door behind itself. Visitors stay
anonymous — saved schemes and the wishlist hang off a browser-generated id and
none of this applies to them.

**Analytics** — what people put in their rooms, not just how many hits: a
session funnel from opening a room to sending an enquiry, the products actually
applied ranked by how often, the rooms that earn their place, and which
surfaces get changed. Events are batched in the browser and flushed with
`sendBeacon` on the way out, and only ever aggregated on read.

---

## Layout

```
server/                Express + SQLite + sharp
  src/db.js            schema, migrations, row hydration
  src/routes/          vendor, rooms, products, share, visitor (saved/wishlist/stores)
  src/storage.js       upload normalisation (photos capped at 2400px, faces squared)
  src/generate.js      procedural demo rooms, panoramas and tile textures
  src/seed.js          seeds the catalogue

server/
  src/auth.js          users, scrypt hashes, session tokens, route guards
  src/materials.js     which model a material uses; paint swatch generation
  src/routes/auth.js   sign-in, bootstrap, accounts
  src/routes/analytics.js  event intake and aggregation

web/
  src/engine/          the renderer — this is where the product lives
    homography.js      4-point solve, inverse, lens distortion both ways
    tileCore.glsl.js   shared GLSL: bond patterns, joints, the five material models, relighting
    surfaceShader.js   two geometry front-ends (homography, ray/plane)
    meshShader.js      the third: box projection over a real mesh
    Renderer.js        flat-photo passes, render targets, export
    Renderer360.js     panorama compositing, sphere viewer, gyroscope
    Renderer3D.js      glTF scene, orbit controls, mesh tagging
    masks.js           polygon rasterising, hit-testing, reference luminance, area, marker anchors
    layouts.js         bond patterns, tile sizes, grout presets, surface types, materials
  src/i18n/            24 locales, lazily loaded, English as the fallback
  src/kiosk.js         kiosk mode and its idle reset
  src/pages/           RoomSelect, Visualizer, Studio, Studio3D, Admin
  src/lib/pdf.js       room sheet, catalogue and spec-sheet PDFs
  src/lib/qr.js        dependency-free QR encoder
```

### Why plain three.js rather than react-three-fiber

Both pipelines are a handful of full-screen quad passes into render targets. A
scene graph adds nothing to that and gets in the way of the multi-target
compositing; a small imperative renderer class keeps uniform updates cheap
enough that dragging the tile-size slider stays smooth.

---

## Using it

### Add your own room photograph

1. **Rooms → Upload your room**. The floor and walls are detected automatically
   and you land straight in the visualizer. Everything below is only needed
   when you want to correct that, or when detection finds nothing.
2. In the Studio, hit **Auto-detect**, or **+ Add** a surface and pick its type.
3. **Perspective** mode: drag the four corners onto a rectangle whose real size
   you know, and enter that size. The blue half-metre grid tells you when you
   have it — when the squares look square on the floor, the perspective is right.
4. **Mask** mode: click to trace the surface, `Enter` to close. Switch to
   **Cut out** and trace anything standing in front of it.
5. **Preview** lays a test tile so you can check the joints against the photo.
6. **Save**, then **Open**.

Straight-on shots with visible floor edges and even lighting work best. Very
wide-angle photos have barrel distortion that a homography cannot model — the
tiles will bow slightly near the frame edge.

### Add a 3D room

1. **Admin → Rooms → Add a 3D room** and pick a `.glb` or `.gltf` file.
2. Tag the meshes a customer may re-tile. Names like `Floor` or `Wall_Back` are
   guessed for you; so is anything large, flat and lying down or standing up.
3. The preview lays a test tile at real-world scale. If a 600 mm tile looks
   like a mosaic, the model was exported in centimetres — re-export it in
   metres rather than scaling it here, because world metres are what every
   size control in the app means.

### Add products

**Admin → Products → New product**, or **Bulk upload** to turn a folder of tile
images into one product each. Pick the **material** first: it decides how the
product is laid, and moves the default size onto something sane for it. Upload
3–6 faces per design for realistic variation (a sheet material wants one
tile-able image of a single repeat; paint wants no image at all, only a
colour), tag which surfaces each product may be applied to, and fill in price
and box quantities so the calculator can do its job.

### Embed it

**Admin → Branding** gives you an iframe snippet, and every visitor-facing
feature has its own on/off switch there. The same page generates a QR code for
any single room, and the kiosk link for a screen on the shop floor.

---

## Notes and limits

- Requires **WebGL 2** (every current browser; the app says so plainly if not).
- Room photos are resized once, on upload, to at most 2400 px. Masks and
  homographies are stored in that pixel space, so re-uploading a photo means
  re-authoring its surfaces.
- Uploads live in `server/data/uploads`, the database in
  `server/data/visualizer.db`. Both are plain files — back them up together.
- The demo rooms are rendered from a real pinhole camera and the 360 rooms by
  ray-casting a box, so the surface geometry that ships with them is exact
  rather than hand-nudged. They are placeholders: replace them with real
  photography.
- Uploaded glTF models are stored untouched. Re-encoding one would break
  exactly the mesh names the Studio uses to identify surfaces.
- Analytics are counted against the same anonymous browser id the wishlist
  uses. No names, no addresses, no cross-site tracking, and a session is one
  browser tab. `POST /api/analytics/prune` drops events older than a given
  number of days.

### Not built yet

- **Plans, subscriptions and billing** — accounts exist, but there is no
  metering, no SKU or session limits, and nothing to sell them with.
- **Studio and Admin translations.** The visitor-facing app is translated; the
  authoring tools are not, on the grounds that they have an audience of one.
- **A hosted environment map for 3D rooms.** They are lit by a key light and a
  hemisphere, which is honest but flatter than an HDRI would be.
- **Per-viewer analytics.** Everything is aggregated on read by design; there
  is no way to ask what one visitor did, and that is intentional.
