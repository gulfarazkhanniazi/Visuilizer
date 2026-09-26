import { Router } from 'express';
import { db, hydrateRoom } from '../db.js';
import {
  upload, uploadModel, saveRoomPhoto, saveModel, saveDataUrl, removeUpload, nano,
} from '../storage.js';
import { detectSurfaces } from '../segmentation.js';
import { requireAuth, mayEditRoom } from '../auth.js';
import path from 'node:path';
import { UPLOAD_DIR } from '../db.js';

const router = Router();

router.get('/room-categories', (req, res) => {
  const rows = db.prepare(`
    SELECT c.id, c.name, c.sort, COUNT(r.id) AS count
      FROM room_categories c
      LEFT JOIN rooms r ON r.category = c.id AND r.is_custom = 0
     GROUP BY c.id
     ORDER BY c.sort, c.name
  `).all();
  res.json(rows);
});

router.post('/room-categories', requireAuth, (req, res) => {
  const { id, name, sort = 0 } = req.body ?? {};
  if (!name) return res.status(400).json({ error: 'name is required' });
  const key = id || name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  db.prepare(`
    INSERT INTO room_categories (id, name, sort) VALUES (?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET name = excluded.name, sort = excluded.sort
  `).run(key, name, sort);
  res.json({ id: key, name, sort });
});

router.get('/rooms', (req, res) => {
  const { category, owner, includeCustom } = req.query;
  const where = [];
  const params = {};
  if (category && category !== 'all') { where.push('category = @category'); params.category = category; }
  if (owner) {
    where.push('(is_custom = 0 OR owner = @owner)');
    params.owner = owner;
  } else if (includeCustom !== '1') {
    where.push('is_custom = 0');
  }
  const sql = `SELECT * FROM rooms ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
               ORDER BY is_custom DESC, sort, created_at DESC`;
  res.json(db.prepare(sql).all(params).map(hydrateRoom));
});

router.get('/rooms/:id', (req, res) => {
  const room = hydrateRoom(db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id));
  if (!room) return res.status(404).json({ error: 'Room not found' });
  res.json(room);
});

/**
 * Upload a room photo. The room is created with no surfaces -- the studio (or
 * the visitor's own "mark your floor" flow) fills in the quads and masks next.
 */
/**
 * A visitor photographing their own room is a public action; adding a preset
 * room to the business's own library is not. Runs after multer, because until
 * the multipart body is parsed there is no `isCustom` to look at.
 */
const guardPresetUpload = (req, res, next) => (
  req.body?.isCustom === '0' ? requireAuth(req, res, next) : next()
);

router.post('/rooms/upload', upload.single('photo'), guardPresetUpload, async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No photo uploaded' });
    const photo = await saveRoomPhoto(req.file.buffer);
    const id = `room_${nano()}`;
    const isCustom = req.body.isCustom !== '0';

    db.prepare(`
      INSERT INTO rooms (id, name, category, image, thumb, width, height, data, is_custom, owner)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      id,
      req.body.name || 'My room',
      req.body.category || null,
      photo.image,
      photo.thumb,
      photo.width,
      photo.height,
      JSON.stringify({ objectList: [], settings: { blurRadius: 6 } }),
      isCustom ? 1 : 0,
      req.body.owner || null,
    );
    res.status(201).json(hydrateRoom(db.prepare('SELECT * FROM rooms WHERE id = ?').get(id)));
  } catch (e) { next(e); }
});


/**
 * Upload a glTF room.
 *
 * A modelled room has no photograph, so it carries a model file instead and is
 * marked `kind: '3d'`. Which of its meshes are tileable surfaces is decided
 * afterwards in the Studio, by name -- the server never parses the glTF.
 */
router.post('/rooms/upload-model', requireAuth, uploadModel.single('model'), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No model uploaded' });
    if (!/\.(glb|gltf)$/i.test(req.file.originalname ?? '')) {
      return res.status(400).json({ error: 'Upload a .glb or .gltf file' });
    }
    const url = await saveModel(req.file.buffer, req.file.originalname);
    const id = `room_${nano()}`;
    db.prepare(`
      INSERT INTO rooms (id, name, category, image, thumb, width, height, data, is_custom, owner, kind, model)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, '3d', ?)
    `).run(
      id,
      req.body.name || req.file.originalname.replace(/\.[^.]+$/, ''),
      req.body.category || null,
      url,
      null,
      1600,
      1000,
      JSON.stringify({ objectList: [], settings: {} }),
      url,
    );
    res.status(201).json(hydrateRoom(db.prepare('SELECT * FROM rooms WHERE id = ?').get(id)));
  } catch (e) { next(e); }
});

/**
 * Find the floor and walls automatically.
 *
 * Kept separate from the upload so the photo appears instantly and the
 * detection (a few seconds of model inference) can report progress. Anything
 * it produces is ordinary editable geometry -- the Studio treats an
 * auto-detected surface exactly like a hand-drawn one.
 */
router.post('/rooms/:id/auto-detect', async (req, res, next) => {
  try {
    const row = db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id);
    if (!row) return res.status(404).json({ error: 'Room not found' });
    if (!mayEditRoom(req, row)) {
      return res.status(401).json({ error: 'Sign in to change this room', needsAuth: true });
    }
    if (row.kind !== '2d') {
      return res.status(400).json({
        error: row.kind === '360'
          ? 'Panorama rooms are configured by their planes, not by detection'
          : 'Modelled rooms already know their geometry; tag their meshes in the Studio',
      });
    }

    const file = path.join(UPLOAD_DIR, path.basename(row.image));
    const result = await detectSurfaces(file, {
      roomHeight: Number(req.body?.roomHeight) || 2.7,
      hfov: Number(req.body?.hfov) || 70,
      includeCeiling: req.body?.includeCeiling === true,
    });

    if (!result.objectList.length) {
      return res.status(422).json({
        error: 'No floor or walls could be found in this photo. Mark them by hand in the Studio.',
        detected: result.detected,
      });
    }

    const current = JSON.parse(row.data);
    db.prepare('UPDATE rooms SET data = ? WHERE id = ?').run(
      JSON.stringify({
        objectList: result.objectList,
        settings: { ...current.settings, camera: result.camera, autoDetected: true },
      }),
      req.params.id,
    );

    res.json({
      room: hydrateRoom(db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id)),
      camera: result.camera,
      detected: result.detected,
    });
  } catch (e) { next(e); }
});

/** Save the surfaces the studio authored for a room. */
router.put('/rooms/:id', async (req, res, next) => {
  try {
    const row = db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id);
    if (!row) return res.status(404).json({ error: 'Room not found' });
    if (!mayEditRoom(req, row)) {
      return res.status(401).json({ error: 'Sign in to change this room', needsAuth: true });
    }

    const current = JSON.parse(row.data);
    const { name, category, objectList, settings, sort, thumb } = req.body ?? {};

    // A modelled room has no photograph to make a thumbnail from, so the
    // Studio sends a rendered one when it saves.
    let thumbUrl = row.thumb;
    if (thumb?.startsWith('data:')) {
      thumbUrl = await saveDataUrl(thumb, 'roomthumb');
      if (row.thumb && row.thumb !== row.image) await removeUpload(row.thumb);
    }

    db.prepare(`
      UPDATE rooms SET name = ?, category = ?, data = ?, sort = ?, thumb = ? WHERE id = ?
    `).run(
      name ?? row.name,
      category !== undefined ? category : row.category,
      JSON.stringify({
        objectList: objectList ?? current.objectList,
        settings: { ...current.settings, ...(settings ?? {}) },
      }),
      sort ?? row.sort,
      thumbUrl,
      req.params.id,
    );
    res.json(hydrateRoom(db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id)));
  } catch (e) { next(e); }
});

router.delete('/rooms/:id', async (req, res, next) => {
  try {
    const row = db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id);
    if (!row) return res.status(404).json({ error: 'Room not found' });
    if (!mayEditRoom(req, row)) {
      return res.status(401).json({ error: 'Sign in to delete this room', needsAuth: true });
    }
    db.prepare('DELETE FROM rooms WHERE id = ?').run(req.params.id);
    await removeUpload(row.image);
    await removeUpload(row.thumb);
    if (row.model && row.model !== row.image) await removeUpload(row.model);
    res.json({ ok: true });
  } catch (e) { next(e); }
});

/** Duplicate a room, so a preset can be used as the starting point for a new one. */
router.post('/rooms/:id/duplicate', requireAuth, (req, res) => {
  const row = db.prepare('SELECT * FROM rooms WHERE id = ?').get(req.params.id);
  if (!row) return res.status(404).json({ error: 'Room not found' });
  const id = `room_${nano()}`;
  db.prepare(`
    INSERT INTO rooms (id, name, category, image, thumb, width, height, data, is_custom, owner, sort)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(id, `${row.name} copy`, row.category, row.image, row.thumb,
         row.width, row.height, row.data, row.is_custom, row.owner, row.sort);
  res.status(201).json(hydrateRoom(db.prepare('SELECT * FROM rooms WHERE id = ?').get(id)));
});

export default router;
