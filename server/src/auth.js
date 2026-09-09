/**
 * Admin authentication.
 *
 * The admin panel used to be open to anyone who could reach `/admin`, which is
 * fine on a laptop and not fine anywhere else -- it can delete the catalogue.
 *
 * Deliberately small: a users table, scrypt password hashes, and opaque
 * session tokens in the database. No JWT, because a token that can be revoked
 * by deleting a row is exactly what a "sign this person out" button needs, and
 * nothing here is distributed enough to pay for stateless verification.
 *
 * Visitors are a separate idea entirely and stay anonymous: their saved schemes
 * and wishlist hang off the browser-generated `x-visitor` id, and none of this
 * applies to them.
 */
import crypto from 'node:crypto';
import { db } from './db.js';

const SESSION_DAYS = 30;
const SCRYPT = { N: 16384, r: 8, p: 1, keylen: 64 };

db.exec(`
CREATE TABLE IF NOT EXISTS users (
  id          TEXT PRIMARY KEY,
  email       TEXT NOT NULL UNIQUE COLLATE NOCASE,
  name        TEXT,
  password    TEXT NOT NULL,
  role        TEXT NOT NULL DEFAULT 'admin',
  active      INTEGER NOT NULL DEFAULT 1,
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  last_login  TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  token      TEXT PRIMARY KEY,
  user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  user_agent TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
`);

/* ------------------------------------------------------------ passwords -- */

export function hashPassword(plain) {
  const salt = crypto.randomBytes(16);
  const key = crypto.scryptSync(String(plain), salt, SCRYPT.keylen, SCRYPT);
  return `scrypt$${salt.toString('hex')}$${key.toString('hex')}`;
}

export function verifyPassword(plain, stored) {
  const [scheme, saltHex, keyHex] = String(stored ?? '').split('$');
  if (scheme !== 'scrypt' || !saltHex || !keyHex) return false;
  const key = crypto.scryptSync(String(plain), Buffer.from(saltHex, 'hex'), SCRYPT.keylen, SCRYPT);
  const expected = Buffer.from(keyHex, 'hex');
  // Lengths must match before timingSafeEqual, which throws otherwise.
  return key.length === expected.length && crypto.timingSafeEqual(key, expected);
}

/* --------------------------------------------------------------- users --- */

export const userCount = () => db.prepare('SELECT COUNT(*) AS n FROM users').get().n;

const publicUser = (row) => (row ? {
  id: row.id,
  email: row.email,
  name: row.name,
  role: row.role,
  active: !!row.active,
  createdAt: row.created_at,
  lastLogin: row.last_login,
} : null);

export function createUser({ email, password, name, role = 'admin' }) {
  const clean = String(email ?? '').trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(clean)) {
    throw Object.assign(new Error('A valid email address is required'), { status: 400 });
  }
  if (String(password ?? '').length < 8) {
    throw Object.assign(new Error('The password must be at least 8 characters'), { status: 400 });
  }
  if (db.prepare('SELECT 1 FROM users WHERE email = ?').get(clean)) {
    throw Object.assign(new Error('That email address already has an account'), { status: 409 });
  }
  const id = `u_${crypto.randomBytes(8).toString('hex')}`;
  db.prepare(`
    INSERT INTO users (id, email, name, password, role) VALUES (?, ?, ?, ?, ?)
  `).run(id, clean, name ?? null, hashPassword(password), role === 'editor' ? 'editor' : 'admin');
  return publicUser(db.prepare('SELECT * FROM users WHERE id = ?').get(id));
}

export function listUsers() {
  return db.prepare('SELECT * FROM users ORDER BY created_at').all().map(publicUser);
}

/**
 * Seed the first administrator from the environment.
 *
 * Lets a container start up already secured. Without it the first person to
 * find the URL gets to claim the account, which is the right trade for a
 * laptop and the wrong one for a deployment.
 */
export function seedFromEnv() {
  const email = process.env.ADMIN_EMAIL;
  const password = process.env.ADMIN_PASSWORD;
  if (!email || !password || userCount() > 0) return null;
  try {
    const user = createUser({ email, password, name: 'Administrator' });
    console.log(`  admin    seeded ${user.email} from ADMIN_EMAIL`);
    return user;
  } catch (e) {
    console.error(`  admin    could not seed from ADMIN_EMAIL: ${e.message}`);
    return null;
  }
}

/* ------------------------------------------------------------ sessions --- */

export function createSession(userId, userAgent) {
  const token = crypto.randomBytes(32).toString('hex');
  const expires = new Date(Date.now() + SESSION_DAYS * 86400_000).toISOString();
  db.prepare(`
    INSERT INTO sessions (token, user_id, user_agent, expires_at) VALUES (?, ?, ?, ?)
  `).run(token, userId, (userAgent ?? '').slice(0, 200), expires);
  db.prepare("DELETE FROM sessions WHERE expires_at < datetime('now')").run();
  return { token, expiresAt: expires };
}

export const destroySession = (token) =>
  db.prepare('DELETE FROM sessions WHERE token = ?').run(token);

function tokenFrom(req) {
  const header = req.get('authorization') ?? '';
  const m = /^Bearer\s+(\S+)$/i.exec(header);
  return m ? m[1] : (req.get('x-admin-token') || null);
}

/** The signed-in user for this request, or null. Never throws. */
export function currentUser(req) {
  const token = tokenFrom(req);
  if (!token) return null;
  const row = db.prepare(`
    SELECT u.* FROM sessions s
      JOIN users u ON u.id = s.user_id
     WHERE s.token = ? AND s.expires_at > datetime('now') AND u.active = 1
  `).get(token);
  return row ? { ...publicUser(row), token } : null;
}

/* ---------------------------------------------------------- middleware --- */

/**
 * Gate a route behind a signed-in administrator.
 *
 * Until the first account exists the panel has to stay reachable, or a fresh
 * install would be unusable -- there would be no way in to create the account
 * that lets you in. `/auth/status` tells the client which of the two states it
 * is looking at, and bootstrap closes the door behind itself.
 */
export function requireAuth(req, res, next) {
  if (userCount() === 0) {
    req.user = { id: 'setup', email: null, role: 'admin', setup: true };
    return next();
  }
  const user = currentUser(req);
  if (!user) return res.status(401).json({ error: 'Sign in to do that', needsAuth: true });
  req.user = user;
  return next();
}

export function requireRole(role) {
  return (req, res, next) => requireAuth(req, res, () => {
    if (req.user.role !== role && !req.user.setup) {
      return res.status(403).json({ error: `This needs ${role} access` });
    }
    return next();
  });
}

/**
 * Rooms have two kinds of owner.
 *
 * A preset room belongs to the business and only an administrator may touch
 * it. A room a visitor photographed belongs to that visitor, who has to be
 * able to mark up its surfaces without an account -- that is the whole
 * "upload your own room" flow.
 */
export function mayEditRoom(req, row) {
  if (!row) return false;
  if (userCount() === 0) return true;
  if (currentUser(req)) return true;
  const visitor = req.get('x-visitor');
  return !!(row.is_custom && visitor && row.owner === visitor);
}

/* ----------------------------------------------------- login throttling -- */

const attempts = new Map();
const WINDOW = 15 * 60_000;
const MAX_ATTEMPTS = 10;

export function throttle(key) {
  const now = Date.now();
  const rec = attempts.get(key);
  if (!rec || now - rec.first > WINDOW) {
    attempts.set(key, { first: now, n: 1 });
    return true;
  }
  rec.n += 1;
  return rec.n <= MAX_ATTEMPTS;
}

export const clearThrottle = (key) => attempts.delete(key);
