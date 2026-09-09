/**
 * Material rendering models, server side.
 *
 * The authoritative catalogue lives in web/src/engine/layouts.js because the
 * shader switches on it; the server only needs to know which model a material
 * uses, and it needs it for one reason: a `solid` product (paint) is a colour
 * rather than a photograph, so it has no face to upload and one has to be
 * generated for it.
 */
import sharp from 'sharp';

export const MATERIAL_MODELS = {
  tile: 'module',
  marble: 'module',
  granite: 'module',
  stone: 'module',
  quartz: 'module',
  wood: 'module',
  hardwood: 'module',
  engineered: 'module',
  laminate: 'module',
  vinyl: 'module',
  spc: 'module',
  wpc: 'module',
  'carpet-tile': 'module',
  'wall-panel': 'module',
  wallpaper: 'sheet',
  carpet: 'sheet',
  epoxy: 'sheet',
  paint: 'solid',
  rug: 'piece',
  grout: 'joint',
};

export const materialModel = (material) => MATERIAL_MODELS[material] ?? 'module';

const HEX = /^#?([0-9a-f]{6})$/i;

export function parseHex(hex, fallback = '#eae3d6') {
  const m = HEX.exec(String(hex ?? '').trim()) ?? HEX.exec(fallback);
  return `#${m[1].toLowerCase()}`;
}

/**
 * A flat swatch for a paint product.
 *
 * The renderer takes the colour from a uniform, so this image is never what
 * gets painted onto the wall -- it is the thumbnail and the catalogue swatch,
 * and it keeps a paint product the same shape as every other product rather
 * than a special case threaded through the whole app.
 */
export async function solidSwatch(hex, size = 256) {
  const c = parseHex(hex);
  const r = parseInt(c.slice(1, 3), 16);
  const g = parseInt(c.slice(3, 5), 16);
  const b = parseInt(c.slice(5, 7), 16);
  return sharp({
    create: { width: size, height: size, channels: 3, background: { r, g, b } },
  }).jpeg({ quality: 92 }).toBuffer();
}
