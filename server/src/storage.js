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
 * Bulk catalogue import: one product per image, up to 200 in a request. Its
 * own limit, so the ordinary 12-file cap above stays where it is.
 */
export const uploadBulk = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024, files: 200 },
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
  let meta;
  try {
    meta = await sharp(buffer, { failOn: 'none' }).metadata();
  } catch {
    meta = null;
  }
  if (!meta?.width || !meta?.height) {
    throw Object.assign(new Error('That file is not a supported image'), { status: 400 });
  }

  // metadata() reports the stored (unrotated) size. A phone photo taken in
  // portrait is stored sideways with an EXIF orientation of 5-8, and .rotate()
  // below turns it upright -- so its width and height swap.
  const upright = (meta.orientation ?? 1) >= 5;
  const srcW = upright ? meta.height : meta.width;
  const srcH = upright ? meta.width : meta.height;

  const scale = Math.min(1, maxSide / Math.max(srcW, srcH));
  // The re-encode below drops EXIF, and with it the one thing in it the
  // scanner needs: the lens. Keep the 35 mm-equivalent focal length.
  const focal35 = exifFocal35(meta.exif);
  let width = Math.round(srcW * scale);
  let height = Math.round(srcH * scale);

  const file = `room-${id}.jpg`;
  const thumbFile = `room-${id}-thumb.jpg`;

  const info = await sharp(buffer, { failOn: 'none' }).rotate()
    .resize(width, height, { fit: 'inside' })
    .jpeg({ quality: 92, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, file));
  // Record exactly what was written: everything is authored in its pixels.
  width = info.width;
  height = info.height;

  await sharp(buffer, { failOn: 'none' }).rotate()
    .resize(480, 320, { fit: 'cover' })
    .jpeg({ quality: 78, mozjpeg: true })
    .toFile(path.join(UPLOAD_DIR, thumbFile));

  return {
    id, image: `/uploads/${file}`, thumb: `/uploads/${thumbFile}`, width, height, focal35,
  };
}

/**
 * FocalLengthIn35mmFilm (tag 0xA405) from a raw EXIF block, or null. Only the
 * TIFF structure needed to reach it: header, IFD0, the Exif sub-IFD pointer.
 */
export function exifFocal35(exif) {
  try {
    if (!exif || exif.length < 14) return null;
    const base = exif.indexOf('Exif\0\0') === 0 ? 6 : 0;
    const le = exif.toString('ascii', base, base + 2) === 'II';
    const u16 = (o) => (le ? exif.readUInt16LE(base + o) : exif.readUInt16BE(base + o));
    const u32 = (o) => (le ? exif.readUInt32LE(base + o) : exif.readUInt32BE(base + o));
    if (u16(2) !== 42) return null;
    const findTag = (ifd, tag) => {
      const n = u16(ifd);
      for (let k = 0; k < n; k++) {
        const e = ifd + 2 + k * 12;
        if (u16(e) === tag) return e;
      }
      return null;
    };
    const ptr = findTag(u32(4), 0x8769);
    if (ptr === null) return null;
    const e = findTag(u32(ptr + 8), 0xa405);
    if (e === null) return null;
    const v = u16(e + 8);
    return v >= 8 && v <= 400 ? v : null;
  } catch {
    return null;
  }
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
