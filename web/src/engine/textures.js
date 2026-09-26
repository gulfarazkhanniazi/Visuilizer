import * as THREE from 'three';

const imageCache = new Map();

/** Load an image once, cached, CORS-enabled so it can go into a texture. */
export function loadImage(url) {
  if (imageCache.has(url)) return imageCache.get(url);
  const p = new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
  imageCache.set(url, p);
  return p;
}

/**
 * Stack a product's faces into one vertical atlas.
 *
 * Real tile ranges ship several "random faces" so a laid floor does not repeat
 * visibly; the shader picks a face per tile by hashing the tile id, which needs
 * them in a single texture. Faces are resampled to a common cell so the atlas
 * stays a clean grid.
 */
export async function buildTileAtlas(urls, maxCell = 1024) {
  const images = await Promise.all(urls.map(loadImage));
  if (!images.length) throw new Error('buildTileAtlas needs at least one image');

  const cellW = Math.min(maxCell, Math.max(...images.map((i) => i.naturalWidth)));
  const cellH = Math.min(maxCell, Math.max(...images.map((i) => i.naturalHeight)));

  const canvas = document.createElement('canvas');
  canvas.width = cellW;
  canvas.height = cellH * images.length;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  images.forEach((img, i) => ctx.drawImage(img, 0, i * cellH, cellW, cellH));

  const texture = new THREE.CanvasTexture(canvas);
  // Faces are stacked vertically, so V must clamp or neighbouring faces bleed
  // into each other; U repeats freely because a tile face tiles horizontally.
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = 16;
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.needsUpdate = true;

  return { texture, faces: images.length, cellW, cellH };
}

/** Wrap a canvas (a rasterised mask) as a clamped, unfiltered-edge texture. */
export function canvasTexture(canvas, { mipmaps = false } = {}) {
  const t = new THREE.CanvasTexture(canvas);
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.minFilter = mipmaps ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = mipmaps;
  t.colorSpace = THREE.LinearSRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

export function imageTexture(img) {
  const t = new THREE.Texture(img);
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.colorSpace = THREE.LinearSRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

/** Read a loaded image back as RGBA bytes, for luminance analysis. */
export function imagePixels(img, maxSide = 900) {
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return { data: ctx.getImageData(0, 0, w, h).data, width: w, height: h, scale };
}

export function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '#ffffff');
  if (!m) return [1, 1, 1];
  return [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
}
