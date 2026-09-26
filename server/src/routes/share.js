import { Router } from 'express';
import { db } from '../db.js';
import { saveDataUrl, nano } from '../storage.js';

const router = Router();

/**
 * Persist a visualisation so it can be reopened from a short link or a QR code.
 * The payload is the full per-surface state, which is small; the preview image
 * is optional and only used for link previews and the admin list.
 */
router.post('/share', async (req, res, next) => {
  try {
    const { payload, preview } = req.body ?? {};
    if (!payload) return res.status(400).json({ error: 'payload is required' });

    const code = nano().slice(0, 8);
    const previewUrl = preview ? await saveDataUrl(preview, 'share') : null;
    db.prepare('INSERT INTO shares (code, payload, preview) VALUES (?, ?, ?)')
      .run(code, JSON.stringify(payload), previewUrl);

    res.status(201).json({ code, preview: previewUrl });
  } catch (e) { next(e); }
});

router.get('/share/:code', (req, res) => {
  const row = db.prepare('SELECT * FROM shares WHERE code = ?').get(req.params.code);
  if (!row) return res.status(404).json({ error: 'That link has expired or never existed' });
  res.json({ code: row.code, payload: JSON.parse(row.payload), preview: row.preview });
});

export default router;
