import multer from 'multer';
import sharp from 'sharp';
import path from 'node:path';
import fs from 'node:fs/promises';
import { customAlphabet } from 'nanoid';
import { UPLOAD_DIR } from './db.js';

export const nano = customAlphabet('0123456789abcdefghijklmnopqrstuvwxyz', 12);

export const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024, files: 12 },
});

/**
 * A separate limit for 3D rooms: a furnished glTF with its textures packed in
 * runs to tens of megabytes, which is nothing like a photograph and should not
 * push the photo limit up to match.
 */
export const uploadModel = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 80 * 1024 * 1024, files: 1 },
});

/**
 * Normalise an uploaded room photo.
 *
 * Everything downstream -- masks, homographies, the lighting plate -- is stored
 * in the photo's pixel space, so the photo must be resized ONCE, here, before
 * anyone authors against it. Capped at 2400px because beyond that the GPU cost
 * of the full-resolution render targets outweighs any visible gain.
 */
export async function saveRoomPhoto(buffer, { maxSide = 2400 } = {}) {
  const id = nano();
  const base = sharp(buffer, { failOn: 'none' }).rotate();
  const meta = await base.metadata();

  const scale = Math.min(1, maxSide / Math.max(meta.width, meta.height));
  const width = Math.round(meta.width * scale);
  const height = Math.round(meta.height * scale);

  const file = `room-${id}.jpg`;
  const thumbFile = `room-${id}-thumb.jpg`;

  await sharp(buffer, { failOn: 'none' }).rotate()
    .resize(width, height, { fit: 'inside' })
    .jpeg({ quality: 92, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, file));

  await sharp(buffer, { failOn: 'none' }).rotate()
    .resize(480, 320, { fit: 'cover' })
    .jpeg({ quality: 78, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, thumbFile));

  return { id, image: `/uploads/${file}`, thumb: `/uploads/${thumbFile}`, width, height };
}

/**
 * Save one tile face.
 *
 * Faces are squared off to a power-of-two cell so several of them stack into a
 * clean atlas on the client and mipmap without bleeding between rows.
 */
export async function saveTileFace(buffer, { size = 1024 } = {}) {
  const id = nano();
  const file = `tile-${id}.jpg`;
  await sharp(buffer, { failOn: 'none' })
    .resize(size, size, { fit: 'fill' })
    .jpeg({ quality: 92, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, file));
  return `/uploads/${file}`;
}

export async function saveThumb(buffer, prefix = 'thumb') {
  const id = nano();
  const file = `${prefix}-${id}.jpg`;
  await sharp(buffer, { failOn: 'none' })
    .resize(320, 320, { fit: 'cover' })
    .jpeg({ quality: 80, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, file));
  return `/uploads/${file}`;
}

/** Persist a data: URL (the client's rendered preview) as a file. */
export async function saveDataUrl(dataUrl, prefix = 'preview') {
  const m = /^data:image\/(png|jpeg|webp);base64,(.+)$/.exec(dataUrl || '');
  if (!m) return null;
  const id = nano();
  const file = `${prefix}-${id}.jpg`;
  await sharp(Buffer.from(m[2], 'base64'))
    .resize(1280, 1280, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 84, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, file));
  return `/uploads/${file}`;
}

/**
 * Store a glTF binary as-is.
 *
 * Deliberately untouched: re-encoding a model is not like re-encoding a photo,
 * and quantising or re-packing one here would break exactly the mesh names the
 * Studio uses to identify surfaces.
 */
export async function saveModel(buffer, originalName = '') {
  const ext = /\.gltf$/i.test(originalName) ? 'gltf' : 'glb';
  const id = nano();
  const file = `model-${id}.${ext}`;
  await fs.writeFile(path.join(UPLOAD_DIR, file), buffer);
  return `/uploads/${file}`;
}

export async function removeUpload(url) {
  if (!url?.startsWith('/uploads/')) return;
  await fs.unlink(path.join(UPLOAD_DIR, path.basename(url))).catch(() => {});
}
