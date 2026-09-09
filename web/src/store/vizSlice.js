import { createSlice } from '@reduxjs/toolkit';
import { defaultSurfaceState, MATERIAL_BY_KEY, materialModel } from '../engine/layouts.js';

const MAX_HISTORY = 40;

/** A frame is one complete "look": every surface's product and settings. */
function blankFrames(room) {
  const build = () => Object.fromEntries(
    (room?.objectList ?? []).map((o) => [
      o.name,
      { ...defaultSurfaceState(o.product_surface), ...(o.defaults ?? {}) },
    ]),
  );
  return { left: build(), right: build() };
}

const initialState = {
  room: null,
  status: 'idle',
  error: null,
  activeSurface: null,
  activeFrame: 'left',
  compare: false,
  split: 0.5,
  // Each surface is its own: walls in a room are genuinely different planes
  // and are usually specified separately. Turn the link on to change a whole
  // set at once.
  linkSameType: false,
  frames: { left: {}, right: {} },
  view: { zoom: 1, x: 0.5, y: 0.5 },
  past: [],
  future: [],
};

function snapshot(state) {
  return JSON.parse(JSON.stringify(state.frames));
}

function pushHistory(state) {
  state.past.push(snapshot(state));
  if (state.past.length > MAX_HISTORY) state.past.shift();
  state.future = [];
}

/** Which surfaces an edit should touch, honouring the link toggle. */
function targets(state, surfaceName) {
  const room = state.room;
  if (!room) return [];
  if (!state.linkSameType) return [surfaceName];
  const src = room.objectList.find((o) => o.name === surfaceName);
  if (!src) return [surfaceName];
  return room.objectList
    .filter((o) => o.product_surface === src.product_surface)
    .map((o) => o.name);
}

const vizSlice = createSlice({
  name: 'viz',
  initialState,
  reducers: {
    roomLoading(state) {
      state.status = 'loading';
      state.error = null;
    },
    roomLoaded(state, { payload }) {
      state.room = payload;
      state.status = 'ready';
      state.frames = blankFrames(payload);
      state.activeSurface = payload.objectList.find((o) => o.isMain)?.name
        ?? payload.objectList[0]?.name ?? null;
      state.past = [];
      state.future = [];
      state.view = { zoom: 1, x: 0.5, y: 0.5 };
      state.compare = false;
      state.activeFrame = 'left';
    },
    roomFailed(state, { payload }) {
      state.status = 'error';
      state.error = payload;
    },

    setActiveSurface(state, { payload }) {
      state.activeSurface = payload;
    },
    setActiveFrame(state, { payload }) {
      state.activeFrame = payload;
    },
    setLinkSameType(state, { payload }) {
      state.linkSameType = payload;
    },

    /** Merge a patch into one surface's state (and its linked siblings). */
    updateSurface(state, { payload }) {
      const { surface, patch, frame, skipHistory } = payload;
      const f = frame ?? state.activeFrame;
      if (!skipHistory) pushHistory(state);
      for (const name of targets(state, surface ?? state.activeSurface)) {
        const cur = state.frames[f][name];
        if (!cur) continue;
        state.frames[f][name] = { ...cur, ...patch };
      }
    },

    /**
     * Apply a product, snapping the tile size to one the product ships in.
     *
     * Materials do not all lay the same way, so switching from tile to
     * wallpaper or to a rug also has to move the size and bond onto something
     * that means anything for the new model -- a 600 mm herringbone is not a
     * sane starting point for a roll of wallpaper.
     */
    applyProduct(state, { payload }) {
      const { surface, product, frame } = payload;
      const f = frame ?? state.activeFrame;
      const spec = MATERIAL_BY_KEY[product.material];
      const model = materialModel(product.material);
      pushHistory(state);
      for (const name of targets(state, surface ?? state.activeSurface)) {
        const cur = state.frames[f][name];
        if (!cur) continue;
        const sizes = product.sizes ?? [];
        const keep = sizes.find((s) => s.w === cur.tileSize.w && s.h === cur.tileSize.h);
        const changedModel = cur.model !== model;

        const next = {
          ...cur,
          productId: product.id,
          model,
          tileSize: keep ?? sizes[0] ?? spec?.size ?? cur.tileSize,
          gloss: product.gloss ?? cur.gloss,
        };
        if (product.color) next.color = product.color;
        // A bond pattern only exists for modular materials; anything else
        // resolves as a plain grid and the control is hidden.
        if (model !== 'module') next.layout = 'grid';
        else if (changedModel && spec?.layout) next.layout = spec.layout;
        if (model === 'sheet' || model === 'piece' || model === 'solid') {
          next.grout = { ...cur.grout, size: 0 };
        } else if ((cur.grout?.size ?? 0) === 0 && changedModel) {
          next.grout = { ...cur.grout, size: 2 };
        }
        if (model === 'piece' && changedModel) next.offset = { x: 0, y: 0 };
        state.frames[f][name] = next;
      }
    },

    setCompare(state, { payload }) {
      state.compare = payload;
      if (payload) {
        // Start the comparison from the current look so the split reads as
        // "before vs after" rather than "your room vs an empty room".
        state.frames.right = JSON.parse(JSON.stringify(state.frames.left));
        state.activeFrame = 'right';
      } else {
        state.activeFrame = 'left';
      }
    },
    setSplit(state, { payload }) {
      state.split = payload;
    },
    swapFrames(state) {
      pushHistory(state);
      const l = state.frames.left;
      state.frames.left = state.frames.right;
      state.frames.right = l;
    },
    /** Promote one side of a comparison to be the single active look. */
    keepFrame(state, { payload }) {
      pushHistory(state);
      state.frames.left = JSON.parse(JSON.stringify(state.frames[payload]));
      state.compare = false;
      state.activeFrame = 'left';
    },

    setView(state, { payload }) {
      state.view = { ...state.view, ...payload };
    },
    resetView(state) {
      state.view = { zoom: 1, x: 0.5, y: 0.5 };
    },

    resetSurfaces(state) {
      pushHistory(state);
      state.frames = blankFrames(state.room);
    },

    /** Restore a shared link's saved look. */
    hydrateFrames(state, { payload }) {
      if (payload?.frames) state.frames = payload.frames;
      if (payload?.compare !== undefined) state.compare = payload.compare;
      if (payload?.split !== undefined) state.split = payload.split;
      if (payload?.activeSurface) state.activeSurface = payload.activeSurface;
    },

    undo(state) {
      const prev = state.past.pop();
      if (!prev) return;
      state.future.push(snapshot(state));
      state.frames = prev;
    },
    redo(state) {
      const next = state.future.pop();
      if (!next) return;
      state.past.push(snapshot(state));
      state.frames = next;
    },
  },
});

export const {
  roomLoading, roomLoaded, roomFailed,
  setActiveSurface, setActiveFrame, setLinkSameType,
  updateSurface, applyProduct,
  setCompare, setSplit, swapFrames, keepFrame,
  setView, resetView, resetSurfaces, hydrateFrames,
  undo, redo,
} = vizSlice.actions;

export default vizSlice.reducer;

export const selectActiveState = (s) =>
  s.viz.frames[s.viz.activeFrame]?.[s.viz.activeSurface] ?? null;

export const selectActiveObject = (s) =>
  s.viz.room?.objectList.find((o) => o.name === s.viz.activeSurface) ?? null;
