/**
 * Tile layout (bond pattern) catalogue.
 *
 * `id` must match the LAYOUT_* constants inside surfaceShader.js -- the actual
 * geometry lives in the fragment shader so that every pattern stays
 * perspective-correct and antialiased for free.
 *
 * `ratio` is the length:width the pattern assumes. Patterns that need a
 * specific aspect (herringbone, basketweave) work with whatever the product
 * gives them, but look best at the ratio noted here.
 */
export const LAYOUTS = [
  { id: 0, key: 'grid',          name: 'Grid / Stack',      icon: 'grid',        offset: 0 },
  { id: 1, key: 'brick',         name: 'Brick (1/2)',       icon: 'brick',       offset: 0.5 },
  { id: 2, key: 'brick-third',   name: 'Brick (1/3)',       icon: 'brick3',      offset: 1 / 3 },
  { id: 3, key: 'vertical',      name: 'Vertical Stack',    icon: 'vgrid',       offset: 0 },
  { id: 4, key: 'vertical-brick',name: 'Vertical Brick',    icon: 'vbrick',      offset: 0.5 },
  { id: 5, key: 'herringbone',   name: 'Herringbone',       icon: 'herring',     ratio: 2 },
  { id: 6, key: 'basketweave',   name: 'Basketweave',       icon: 'basket',      ratio: 2 },
  { id: 7, key: 'diagonal',      name: 'Diagonal 45°',      icon: 'diag',        rotate: 45 },
  { id: 8, key: 'diagonal-brick',name: 'Diagonal Brick',    icon: 'diagbrick',   rotate: 45, offset: 0.5 },
];

export const LAYOUT_BY_KEY = Object.fromEntries(LAYOUTS.map((l) => [l.key, l]));

/** Common commercial tile sizes, in millimetres. */
export const TILE_SIZES = [
  { w: 100,  h: 100,  label: '100 × 100' },
  { w: 150,  h: 150,  label: '150 × 150' },
  { w: 200,  h: 200,  label: '200 × 200' },
  { w: 250,  h: 400,  label: '250 × 400' },
  { w: 300,  h: 300,  label: '300 × 300' },
  { w: 300,  h: 600,  label: '300 × 600' },
  { w: 400,  h: 400,  label: '400 × 400' },
  { w: 600,  h: 600,  label: '600 × 600' },
  { w: 600,  h: 1200, label: '600 × 1200' },
  { w: 800,  h: 800,  label: '800 × 800' },
  { w: 1200, h: 1200, label: '1200 × 1200' },
  { w: 1200, h: 2400, label: '1200 × 2400' },
  { w: 150,  h: 900,  label: '150 × 900 (plank)' },
  { w: 200,  h: 1200, label: '200 × 1200 (plank)' },
];

/** Grout presets: thickness in mm plus a colour. */
export const GROUT_SIZES = [0, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12];

export const GROUT_COLORS = [
  { name: 'Bright White', hex: '#f5f5f2' },
  { name: 'Silver',       hex: '#c9c9c4' },
  { name: 'Pearl Grey',   hex: '#a8a8a3' },
  { name: 'Cement',       hex: '#8d8d87' },
  { name: 'Charcoal',     hex: '#4a4a48' },
  { name: 'Black',        hex: '#1c1c1c' },
  { name: 'Sand',         hex: '#cfc0a5' },
  { name: 'Beige',        hex: '#b9a88c' },
  { name: 'Chocolate',    hex: '#5b463a' },
  { name: 'Terracotta',   hex: '#a5613f' },
];

/** Surfaces a room can expose, mirroring the industry's standard vocabulary. */
export const SURFACE_TYPES = [
  { key: 'floor',      name: 'Floor',       plane: 'horizontal' },
  { key: 'wall',       name: 'Wall',        plane: 'vertical' },
  { key: 'backsplash', name: 'Backsplash',  plane: 'vertical' },
  { key: 'countertop', name: 'Countertop',  plane: 'horizontal' },
  { key: 'ceiling',    name: 'Ceiling',     plane: 'horizontal' },
  { key: 'riser',      name: 'Step & Riser',plane: 'vertical' },
  { key: 'cabinet',    name: 'Cabinet',     plane: 'vertical' },
  { key: 'outdoor',    name: 'Outdoor',     plane: 'horizontal' },
  { key: 'rug',        name: 'Rug',         plane: 'horizontal' },
  { key: 'wardrobe',   name: 'Wardrobe',    plane: 'vertical' },
  { key: 'waterline',  name: 'Waterline',   plane: 'vertical' },
];

export const SURFACE_BY_KEY = Object.fromEntries(SURFACE_TYPES.map((s) => [s.key, s]));

/** Default per-surface render state, matching what the shader expects. */
export function defaultSurfaceState(surfaceKey = 'floor') {
  return {
    productId: null,
    // The rendering model of the applied product, mirrored into the surface
    // state so the panels can pick their controls without a product lookup.
    model: 'module',
    tileSize: surfaceKey === 'floor' ? { w: 600, h: 600 } : { w: 300, h: 600 },
    layout: 'grid',
    rotation: 0,          // degrees, applied before the layout
    offset: { x: 0, y: 0 },// metres
    grout: { size: 2, color: '#c9c9c4' },
    bevel: 0.35,          // joint shading strength
    gloss: 0.25,          // 0 matt .. 1 polished
    shade: 1.0,           // how strongly the room's own light/shadow is kept
    detail: 0.6,          // how much fine photo detail bleeds through
    randomFace: true,
    randomRotate: false,
    tint: '#ffffff',
    // Only the `solid` model reads this; it is the paint colour itself rather
    // than a tint over a texture, so it has to survive a product change.
    color: '#eae3d6',
    visible: true,
  };
}

/* ------------------------------------------------------------ materials -- */

/**
 * Rendering models.
 *
 * `material` used to be a label on the product and nothing else -- every
 * material rendered as a grouted tile grid, which is right for tile and wrong
 * for paint, a rug or a roll of wallpaper. Each model below is a different way
 * of putting a product onto a plane, and the id is what the fragment shader
 * switches on (see MODEL_* in tileCore.glsl.js).
 *
 *   module  discrete units with joints between them -- tile, stone, plank,
 *           carpet tile, wall panel. The original behaviour.
 *   sheet   one continuous pattern repeating over the plane, no joints at all
 *           -- wallpaper, broadloom carpet, poured epoxy.
 *   solid   a flat colour with a sheen and no texture -- paint.
 *   piece   a single bounded rectangle laid on the plane, the rest of the
 *           surface left as it was photographed -- a rug.
 *   joint   recolours only the joints of the surface already in the photo,
 *           leaving its tile faces alone -- the grout visualizer.
 */
export const MODELS = {
  module: 0,
  sheet: 1,
  solid: 2,
  piece: 3,
  joint: 4,
};

export const MATERIALS = [
  { key: 'tile',        name: 'Tile',              model: 'module', size: { w: 600, h: 600 } },
  { key: 'marble',      name: 'Marble',            model: 'module', size: { w: 600, h: 1200 } },
  { key: 'granite',     name: 'Granite',           model: 'module', size: { w: 600, h: 600 } },
  { key: 'stone',       name: 'Stone',             model: 'module', size: { w: 400, h: 400 } },
  { key: 'quartz',      name: 'Quartz',            model: 'module', size: { w: 600, h: 1200 } },
  { key: 'wood',        name: 'Wooden flooring',   model: 'module', size: { w: 200, h: 1200 }, layout: 'brick-third' },
  { key: 'hardwood',    name: 'Hardwood',          model: 'module', size: { w: 150, h: 900 },  layout: 'brick-third' },
  { key: 'engineered',  name: 'Engineered wood',   model: 'module', size: { w: 190, h: 1900 }, layout: 'brick-third' },
  { key: 'laminate',    name: 'Laminate',          model: 'module', size: { w: 195, h: 1380 }, layout: 'brick-third' },
  { key: 'vinyl',       name: 'Vinyl',             model: 'module', size: { w: 185, h: 1220 }, layout: 'brick-third' },
  { key: 'spc',         name: 'SPC',               model: 'module', size: { w: 180, h: 1220 }, layout: 'brick-third' },
  { key: 'wpc',         name: 'WPC',               model: 'module', size: { w: 180, h: 1220 }, layout: 'brick-third' },
  { key: 'carpet-tile', name: 'Carpet tile',       model: 'module', size: { w: 500, h: 500 } },
  { key: 'wall-panel',  name: 'Wall panel',        model: 'module', size: { w: 600, h: 2400 }, layout: 'vertical' },
  { key: 'wallpaper',   name: 'Wallpaper',         model: 'sheet',  size: { w: 530, h: 1000 } },
  { key: 'carpet',      name: 'Carpet (broadloom)',model: 'sheet',  size: { w: 900, h: 900 } },
  { key: 'epoxy',       name: 'Epoxy flooring',    model: 'sheet',  size: { w: 2000, h: 2000 } },
  { key: 'paint',       name: 'Paint',             model: 'solid' },
  { key: 'rug',         name: 'Rug',               model: 'piece',  size: { w: 2000, h: 1400 } },
  { key: 'grout',       name: 'Grout only',        model: 'joint',  size: { w: 600, h: 600 } },
];

export const MATERIAL_BY_KEY = Object.fromEntries(MATERIALS.map((m) => [m.key, m]));

/** The rendering model a product uses. Unknown materials behave like tile. */
export function materialModel(material) {
  return MATERIAL_BY_KEY[material]?.model ?? 'module';
}

export function materialModelId(material) {
  return MODELS[materialModel(material)] ?? 0;
}

/** Which of the viewer's control groups make sense for a model. */
export const MODEL_CONTROLS = {
  module: { size: true, bond: true, grout: true, variation: true, finish: true, colour: false, place: false },
  sheet:  { size: true, bond: false, grout: false, variation: false, finish: true, colour: false, place: false },
  solid:  { size: false, bond: false, grout: false, variation: false, finish: true, colour: true, place: false },
  piece:  { size: true, bond: false, grout: false, variation: false, finish: true, colour: false, place: true },
  joint:  { size: true, bond: true, grout: true, variation: false, finish: false, colour: false, place: false },
};

/** A ready-made paint card, so "paint" is usable without uploading anything. */
export const PAINT_COLORS = [
  { name: 'Chalk White',    hex: '#f4f2ed' },
  { name: 'Soft Linen',     hex: '#eae3d6' },
  { name: 'Warm Grey',      hex: '#cfc9c0' },
  { name: 'Stone',          hex: '#b3aca2' },
  { name: 'Clay',           hex: '#c09a80' },
  { name: 'Terracotta',     hex: '#b5613f' },
  { name: 'Ochre',          hex: '#c8993f' },
  { name: 'Olive',          hex: '#7f8560' },
  { name: 'Sage',           hex: '#a8b5a2' },
  { name: 'Eucalyptus',     hex: '#6f8b80' },
  { name: 'Teal',           hex: '#376b73' },
  { name: 'Denim',          hex: '#4a6585' },
  { name: 'Navy',           hex: '#2b3a55' },
  { name: 'Plum',           hex: '#6b4a63' },
  { name: 'Charcoal',       hex: '#3d3f42' },
  { name: 'Graphite',       hex: '#22252a' },
];
