/**
 * A small per-address rate limit for the public write endpoints (share links,
 * enquiries, analytics events) -- the ones anyone can call without an account,
 * and so the ones a script could use to fill the disk or the database.
 *
 * The limits are far above anything a real visitor does, so they only ever
 * stop automated abuse. In memory, per process; RATE_LIMIT=off disables it.
 */
export function rateLimit({ windowMs = 10 * 60_000, max = 60, name = 'requests' } = {}) {
  const hits = new Map();
  let lastSweep = Date.now();
  return (req, res, next) => {
    if (process.env.RATE_LIMIT === 'off') return next();
    const now = Date.now();
    // Forget old windows now and then, so the map cannot grow without limit.
    if (now - lastSweep > windowMs) {
      for (const [k, v] of hits) if (now - v.start > windowMs) hits.delete(k);
      lastSweep = now;
    }
    const key = req.ip ?? 'unknown';
    const rec = hits.get(key);
    if (!rec || now - rec.start > windowMs) {
      hits.set(key, { start: now, n: 1 });
      return next();
    }
    rec.n += 1;
    if (rec.n > max) {
      res.set('Retry-After', String(Math.ceil((rec.start + windowMs - now) / 1000)));
      return res.status(429).json({ error: `Too many ${name}. Please try again in a few minutes.` });
    }
    return next();
  };
}
