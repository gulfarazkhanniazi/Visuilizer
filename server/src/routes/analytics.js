import { Router } from 'express';
import { db } from '../db.js';
import { requireAuth } from '../auth.js';

/**
 * Usage analytics.
 *
 * The point of a visualizer on a tile merchant's site is to find out what
 * people put in their rooms and then never bought, so this records what was
 * applied to what, not just page views. Events are appended by the browser in
 * small batches and only ever aggregated on read -- there is no per-visitor
 * report, and nothing here is joined back to an identity.
 *
 * The visitor id is the same anonymous browser-generated one the wishlist
 * uses. It is not tied to a person, and it is only ever counted, never listed.
 */
db.exec(`
CREATE TABLE IF NOT EXISTS events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  type       TEXT NOT NULL,
  visitor    TEXT,
  session    TEXT,
  room_id    TEXT,
  product_id TEXT,
  surface    TEXT,
  meta       TEXT,
  day        TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, day);
CREATE INDEX IF NOT EXISTS idx_events_product ON events(product_id);
`);

const router = Router();

// Everything the client is allowed to record. An allowlist rather than free
// text, so a stray call cannot quietly invent a metric nobody can interpret.
const TYPES = new Set([
  'session_start', 'room_view', 'room_upload', 'auto_detect',
  'product_apply', 'surface_select', 'compare', 'calculator', 'measure',
  'save_scheme', 'wishlist_add', 'share', 'download', 'pdf', 'lead',
  'view_360', 'studio_save', 'kiosk_reset',
]);

const insert = db.prepare(`
  INSERT INTO events (type, visitor, session, room_id, product_id, surface, meta, day)
  VALUES (@type, @visitor, @session, @roomId, @productId, @surface, @meta, date('now'))
`);
const insertMany = db.transaction((rows) => rows.forEach((r) => insert.run(r)));

router.post('/events', (req, res) => {
  const batch = Array.isArray(req.body?.events) ? req.body.events.slice(0, 50) : [];
  // sendBeacon cannot set headers, so an unloading page passes the visitor id
  // in the body instead. Same anonymous id either way.
  const visitor = req.get('x-visitor')
    ?? (req.body?.visitor ? String(req.body.visitor).slice(0, 64) : null);
  const rows = batch
    .filter((e) => TYPES.has(e?.type))
    .map((e) => ({
      type: e.type,
      visitor,
      session: e.session ? String(e.session).slice(0, 64) : null,
      roomId: e.roomId ? String(e.roomId).slice(0, 64) : null,
      productId: e.productId ? String(e.productId).slice(0, 64) : null,
      surface: e.surface ? String(e.surface).slice(0, 64) : null,
      meta: e.meta ? JSON.stringify(e.meta).slice(0, 400) : null,
    }));

  if (rows.length) insertMany(rows);
  // 204 rather than a body: the client fires these with sendBeacon and is not
  // listening for an answer.
  res.status(204).end();
});

const scalar = (sql, params) => db.prepare(sql).get(params)?.n ?? 0;

router.get('/analytics', requireAuth, (req, res) => {
  const days = Math.min(365, Math.max(1, Number(req.query.days) || 30));
  const since = `-${days - 1} days`;
  const p = { since };

  const where = "day >= date('now', @since)";

  const totals = {
    visitors: scalar(`SELECT COUNT(DISTINCT visitor) AS n FROM events WHERE ${where} AND visitor IS NOT NULL`, p),
    sessions: scalar(`SELECT COUNT(DISTINCT session) AS n FROM events WHERE ${where} AND session IS NOT NULL`, p),
    events: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where}`, p),
    roomViews: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type = 'room_view'`, p),
    uploads: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type = 'room_upload'`, p),
    applies: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type = 'product_apply'`, p),
    shares: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type = 'share'`, p),
    downloads: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type IN ('download','pdf')`, p),
    saves: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type = 'save_scheme'`, p),
    calculators: scalar(`SELECT COUNT(*) AS n FROM events WHERE ${where} AND type = 'calculator'`, p),
    leads: scalar(`SELECT COUNT(*) AS n FROM leads WHERE date(created_at) >= date('now', @since)`, p),
  };

  // One row per day in the window, including the days nothing happened --
  // a gap in a series has to read as a zero, not as a missing bar.
  const raw = db.prepare(`
    SELECT day,
           COUNT(DISTINCT session) AS sessions,
           SUM(type = 'room_view') AS views,
           SUM(type = 'product_apply') AS applies,
           SUM(type IN ('share','download','pdf','save_scheme','lead')) AS intents
      FROM events
     WHERE ${where}
     GROUP BY day
  `).all(p);
  const byDay = Object.fromEntries(raw.map((r) => [r.day, r]));

  const daily = [];
  for (let i = days - 1; i >= 0; i--) {
    const day = db.prepare("SELECT date('now', ?) AS d").get(`-${i} days`).d;
    const r = byDay[day];
    daily.push({
      day,
      sessions: r?.sessions ?? 0,
      views: r?.views ?? 0,
      applies: r?.applies ?? 0,
      intents: r?.intents ?? 0,
    });
  }

  const topProducts = db.prepare(`
    SELECT e.product_id AS id, p.name, p.thumb, p.material, COUNT(*) AS n,
           COUNT(DISTINCT e.session) AS sessions
      FROM events e LEFT JOIN products p ON p.id = e.product_id
     WHERE ${where} AND e.type = 'product_apply' AND e.product_id IS NOT NULL
     GROUP BY e.product_id
     ORDER BY n DESC LIMIT 12
  `).all(p);

  const topRooms = db.prepare(`
    SELECT e.room_id AS id, r.name, r.thumb, COUNT(*) AS n
      FROM events e LEFT JOIN rooms r ON r.id = e.room_id
     WHERE ${where} AND e.type = 'room_view' AND e.room_id IS NOT NULL
     GROUP BY e.room_id
     ORDER BY n DESC LIMIT 10
  `).all(p);

  const topSurfaces = db.prepare(`
    SELECT surface AS id, COUNT(*) AS n
      FROM events
     WHERE ${where} AND type = 'product_apply' AND surface IS NOT NULL
     GROUP BY surface ORDER BY n DESC LIMIT 10
  `).all(p);

  // Where people stop. Each step is counted in sessions, not events, so a
  // visitor who applied forty tiles counts once.
  const step = (sql) => scalar(
    `SELECT COUNT(DISTINCT session) AS n FROM events WHERE ${where} AND session IS NOT NULL AND ${sql}`, p,
  );
  const funnel = [
    { key: 'Opened a room', n: step("type = 'room_view'") },
    { key: 'Applied a product', n: step("type = 'product_apply'") },
    { key: 'Priced it up', n: step("type IN ('calculator','measure')") },
    { key: 'Kept it', n: step("type IN ('save_scheme','download','pdf','share')") },
    { key: 'Got in touch', n: step("type = 'lead'") },
  ];

  res.json({ days, totals, daily, topProducts, topRooms, topSurfaces, funnel });
});

/** Housekeeping, so the events table cannot grow without limit. */
router.post('/analytics/prune', requireAuth, (req, res) => {
  const keep = Math.min(1095, Math.max(30, Number(req.body?.keepDays) || 400));
  const info = db.prepare("DELETE FROM events WHERE day < date('now', ?)").run(`-${keep} days`);
  res.json({ deleted: info.changes, keepDays: keep });
});

export default router;
