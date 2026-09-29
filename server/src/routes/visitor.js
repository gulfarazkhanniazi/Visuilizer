import { Router } from 'express';
import { db, hydrateProduct } from '../db.js';
import { saveDataUrl, removeUpload, nano } from '../storage.js';
import { requireAuth } from '../auth.js';

const router = Router();

/**
 * Everything keyed to a visitor id rather than a login: saved room schemes and
 * a product wishlist. The id is generated in the browser and sent as a header,
 * which is enough for "come back to what I was looking at" without asking
 * anyone to create an account.
 */
function owner(req) {
  const id = req.get('x-visitor') || req.query.owner || req.body?.owner;
  if (!id) {
    const err = new Error('A visitor id is required');
    err.status = 400;
    throw err;
  }
  return String(id);
}

/* ----------------------------------------------------------- saved rooms -- */

router.get('/saved-rooms', (req, res, next) => {
  try {
    const rows = db.prepare(`
      SELECT s.*, r.name AS room_name, r.thumb AS room_thumb, r.kind AS room_kind
        FROM saved_rooms s
        LEFT JOIN rooms r ON r.id = s.room_id
       WHERE s.owner = ?
       ORDER BY s.created_at DESC
    `).all(owner(req));
    res.json(rows.map((r) => ({
      id: r.id,
      roomId: r.room_id,
      roomName: r.room_name,
      roomThumb: r.room_thumb,
      roomKind: r.room_kind,
      name: r.name,
      payload: JSON.parse(r.payload),
      preview: r.preview,
      createdAt: r.created_at,
    })));
  } catch (e) { next(e); }
});

router.post('/saved-rooms', async (req, res, next) => {
  try {
    const who = owner(req);
    const { roomId, name, payload, preview } = req.body ?? {};
    if (!roomId || !payload) {
      return res.status(400).json({ error: 'roomId and payload are required' });
    }
    const id = `sr_${nano()}`;
    const previewUrl = preview ? await saveDataUrl(preview, 'saved') : null;
    db.prepare(`
      INSERT INTO saved_rooms (id, owner, room_id, name, payload, preview)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(id, who, roomId, name || 'My scheme', JSON.stringify(payload), previewUrl);
    res.status(201).json({ id, preview: previewUrl });
  } catch (e) { next(e); }
});

router.delete('/saved-rooms/:id', async (req, res, next) => {
  try {
    const row = db.prepare('SELECT * FROM saved_rooms WHERE id = ? AND owner = ?')
      .get(req.params.id, owner(req));
    if (!row) return res.status(404).json({ error: 'Not found' });
    db.prepare('DELETE FROM saved_rooms WHERE id = ?').run(req.params.id);
    await removeUpload(row.preview);
    res.json({ ok: true });
  } catch (e) { next(e); }
});

/* -------------------------------------------------------------- wishlist -- */

router.get('/wishlist', (req, res, next) => {
  try {
    const rows = db.prepare(`
      SELECT p.* FROM wishlist w
        JOIN products p ON p.id = w.product_id
       WHERE w.owner = ?
       ORDER BY w.created_at DESC
    `).all(owner(req));
    res.json(rows.map(hydrateProduct));
  } catch (e) { next(e); }
});

router.post('/wishlist/:productId', (req, res, next) => {
  try {
    db.prepare(`
      INSERT INTO wishlist (owner, product_id) VALUES (?, ?)
      ON CONFLICT(owner, product_id) DO NOTHING
    `).run(owner(req), req.params.productId);
    res.status(201).json({ ok: true });
  } catch (e) { next(e); }
});

router.delete('/wishlist/:productId', (req, res, next) => {
  try {
    db.prepare('DELETE FROM wishlist WHERE owner = ? AND product_id = ?')
      .run(owner(req), req.params.productId);
    res.json({ ok: true });
  } catch (e) { next(e); }
});

/* ---------------------------------------------------------------- stores -- */

router.get('/stores', (req, res) => {
  res.json(db.prepare('SELECT * FROM stores ORDER BY sort, name').all());
});

router.post('/stores', requireAuth, (req, res) => {
  const b = req.body ?? {};
  if (!b.name) return res.status(400).json({ error: 'name is required' });
  const id = b.id || `st_${nano()}`;
  db.prepare(`
    INSERT INTO stores (id, name, address, city, phone, email, lat, lng, sort)
    VALUES (@id, @name, @address, @city, @phone, @email, @lat, @lng, @sort)
    ON CONFLICT(id) DO UPDATE SET
      name = excluded.name, address = excluded.address, city = excluded.city,
      phone = excluded.phone, email = excluded.email,
      lat = excluded.lat, lng = excluded.lng, sort = excluded.sort
  `).run({
    id,
    name: b.name,
    address: b.address ?? null,
    city: b.city ?? null,
    phone: b.phone ?? null,
    email: b.email ?? null,
    // An empty field means "no location", not 0,0 off the coast of Africa.
    lat: b.lat != null && b.lat !== '' ? Number(b.lat) : null,
    lng: b.lng != null && b.lng !== '' ? Number(b.lng) : null,
    sort: Number(b.sort ?? 0),
  });
  res.json(db.prepare('SELECT * FROM stores WHERE id = ?').get(id));
});

router.delete('/stores/:id', requireAuth, (req, res) => {
  db.prepare('DELETE FROM stores WHERE id = ?').run(req.params.id);
  res.json({ ok: true });
});

export default router;
