import { Router } from 'express';
import { db } from '../db.js';
import { upload, saveThumb } from '../storage.js';
import { requireAuth } from '../auth.js';
import { rateLimit } from '../ratelimit.js';

// Enquiry fields are free text from anyone; keep them to sensible sizes.
const cap = (v, n) => (v == null ? null : String(v).slice(0, n));
// The scheme the visitor was looking at. Never cut a JSON string -- the lead
// list parses it -- so an implausibly large one is replaced, not truncated.
const contextJson = (context) => {
  if (!context) return null;
  const s = JSON.stringify(context);
  return s.length <= 50000 ? s : JSON.stringify({ omitted: 'context too large' });
};

const router = Router();

function readVendor() {
  const row = db.prepare('SELECT * FROM vendor WHERE id = 1').get();
  return {
    name: row.name,
    logo: row.logo,
    primaryColor: row.primary_color,
    settings: JSON.parse(row.settings),
  };
}

router.get('/vendor', (req, res) => res.json(readVendor()));

router.put('/vendor', requireAuth, (req, res) => {
  const current = readVendor();
  const { name, primaryColor, settings } = req.body ?? {};
  db.prepare(`
    UPDATE vendor
       SET name = ?, primary_color = ?, settings = ?, updated_at = datetime('now')
     WHERE id = 1
  `).run(
    name ?? current.name,
    primaryColor ?? current.primaryColor,
    JSON.stringify({ ...current.settings, ...(settings ?? {}) }),
  );
  res.json(readVendor());
});

router.post('/vendor/logo', requireAuth, upload.single('logo'), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No logo uploaded' });
    const url = await saveThumb(req.file.buffer, 'logo');
    db.prepare('UPDATE vendor SET logo = ? WHERE id = 1').run(url);
    res.json(readVendor());
  } catch (e) { next(e); }
});

router.post('/leads', rateLimit({ max: 30, name: 'enquiries' }), (req, res) => {
  const { name, email, phone, message, context } = req.body ?? {};
  if (!email && !phone) {
    return res.status(400).json({ error: 'An email address or phone number is required' });
  }
  const info = db.prepare(`
    INSERT INTO leads (name, email, phone, message, context)
    VALUES (?, ?, ?, ?, ?)
  `).run(cap(name, 200), cap(email, 200), cap(phone, 60), cap(message, 5000),
         contextJson(context));
  res.status(201).json({ id: info.lastInsertRowid });
});

// Enquiries carry visitors' contact details, so reading them needs an account
// even though leaving one does not.
router.get('/leads', requireAuth, (req, res) => {
  const rows = db.prepare('SELECT * FROM leads ORDER BY id DESC LIMIT 500').all();
  res.json(rows.map((r) => ({ ...r, context: r.context ? JSON.parse(r.context) : null })));
});

export default router;
