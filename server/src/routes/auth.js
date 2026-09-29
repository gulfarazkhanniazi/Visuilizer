import { Router } from 'express';
import { db } from '../db.js';
import {
  createUser, listUsers, userCount, verifyPassword, hashPassword,
  createSession, destroySession, currentUser, requireAuth, requireRole,
  throttle, clearThrottle,
} from '../auth.js';

const router = Router();

/**
 * What the client needs before it can show anything: whether this install has
 * been claimed yet, and who -- if anyone -- is signed in.
 */
router.get('/auth/status', (req, res) => {
  res.json({ needsSetup: userCount() === 0, user: currentUser(req) });
});

/** Claim a fresh install. Only ever works once. */
router.post('/auth/bootstrap', (req, res, next) => {
  try {
    if (userCount() > 0) {
      return res.status(409).json({ error: 'This installation already has an administrator' });
    }
    const { email, password, name } = req.body ?? {};
    const user = createUser({ email, password, name, role: 'admin' });
    const session = createSession(user.id, req.get('user-agent'));
    res.status(201).json({ user, ...session });
  } catch (e) { next(e); }
});

router.post('/auth/login', (req, res) => {
  const email = String(req.body?.email ?? '').trim().toLowerCase();
  const password = String(req.body?.password ?? '');
  const key = `${req.ip}|${email}`;

  if (!throttle(key)) {
    return res.status(429).json({ error: 'Too many attempts. Try again in a few minutes.' });
  }

  const row = db.prepare('SELECT * FROM users WHERE email = ? AND active = 1').get(email);
  // One message for both failures, so the response cannot be used to find out
  // which email addresses have accounts.
  if (!row || !verifyPassword(password, row.password)) {
    return res.status(401).json({ error: 'That email address and password do not match' });
  }

  clearThrottle(key);
  db.prepare("UPDATE users SET last_login = datetime('now') WHERE id = ?").run(row.id);
  const session = createSession(row.id, req.get('user-agent'));
  res.json({
    user: { id: row.id, email: row.email, name: row.name, role: row.role },
    ...session,
  });
});

router.post('/auth/logout', (req, res) => {
  const user = currentUser(req);
  if (user) destroySession(user.token);
  res.json({ ok: true });
});

/* ----------------------------------------------------------- user admin -- */

router.get('/auth/users', requireAuth, (req, res) => res.json(listUsers()));

// Creating or removing accounts is an administrator's job: an editor must not
// be able to mint an admin account or delete one.
router.post('/auth/users', requireRole('admin'), (req, res, next) => {
  try {
    const { email, password, name, role } = req.body ?? {};
    res.status(201).json(createUser({ email, password, name, role }));
  } catch (e) { next(e); }
});

router.post('/auth/password', requireAuth, (req, res) => {
  const { currentPassword, newPassword } = req.body ?? {};
  if (String(newPassword ?? '').length < 8) {
    return res.status(400).json({ error: 'The new password must be at least 8 characters' });
  }
  const row = db.prepare('SELECT * FROM users WHERE id = ?').get(req.user.id);
  if (!row) return res.status(404).json({ error: 'Account not found' });
  if (!verifyPassword(currentPassword, row.password)) {
    return res.status(401).json({ error: 'Your current password is not right' });
  }
  db.prepare('UPDATE users SET password = ? WHERE id = ?').run(hashPassword(newPassword), row.id);
  // Every other session for this account is now stale; only this one survives.
  db.prepare('DELETE FROM sessions WHERE user_id = ? AND token != ?').run(row.id, req.user.token);
  res.json({ ok: true });
});

router.delete('/auth/users/:id', requireRole('admin'), (req, res) => {
  if (req.params.id === req.user.id) {
    return res.status(400).json({ error: 'You cannot delete the account you are signed in with' });
  }
  if (userCount() <= 1) {
    return res.status(400).json({ error: 'The last administrator cannot be removed' });
  }
  db.prepare('DELETE FROM sessions WHERE user_id = ?').run(req.params.id);
  db.prepare('DELETE FROM users WHERE id = ?').run(req.params.id);
  res.json({ ok: true });
});

export default router;
