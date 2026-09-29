import express from 'express';
import cors from 'cors';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { UPLOAD_DIR } from './db.js';
import vendorRoutes from './routes/vendor.js';
import roomRoutes from './routes/rooms.js';
import productRoutes from './routes/products.js';
import shareRoutes from './routes/share.js';
import visitorRoutes from './routes/visitor.js';
import authRoutes from './routes/auth.js';
import analyticsRoutes from './routes/analytics.js';
import { seedFromEnv, userCount } from './auth.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 5178;

const app = express();
app.use(cors());
app.use(express.json({ limit: '8mb' }));

// Uploaded photos and tile faces. Immutable filenames, so cache hard.
app.use('/uploads', express.static(UPLOAD_DIR, {
  maxAge: '30d',
  immutable: true,
}));

app.use('/api', authRoutes);
app.use('/api', analyticsRoutes);
app.use('/api', vendorRoutes);
app.use('/api', roomRoutes);
app.use('/api', productRoutes);
app.use('/api', shareRoutes);
app.use('/api', visitorRoutes);

app.get('/api/health', (req, res) => res.json({ ok: true }));

// Serve the built SPA when it exists, so `npm start` runs the whole product.
const dist = path.resolve(__dirname, '..', '..', 'web', 'dist');
app.use(express.static(dist));
app.get(/^\/(?!api|uploads).*/, (req, res, next) => {
  res.sendFile(path.join(dist, 'index.html'), (err) => (err ? next() : null));
});

app.use((err, req, res, next) => {
  console.error(err);
  // An upload the limits rejected is the client's problem, not a server
  // fault: too big is 413, anything else multer refuses is 400.
  if (err.name === 'MulterError') {
    return res.status(err.code === 'LIMIT_FILE_SIZE' ? 413 : 400).json({ error: err.message });
  }
  // Database errors carry table and column names; they stay in the log.
  if (typeof err.code === 'string' && err.code.startsWith('SQLITE_')) {
    return res.status(500).json({ error: 'Server error' });
  }
  res.status(err.status || 500).json({ error: err.message || 'Server error' });
});

seedFromEnv();

const server = app.listen(PORT, () => {
  console.log(`  API      http://localhost:${PORT}/api`);
  console.log(`  uploads  http://localhost:${PORT}/uploads`);
  if (userCount() === 0) {
    console.log('');
    console.log('  The admin panel has no account yet and is open to anyone who');
    console.log('  reaches it. Open /admin to claim it, or set ADMIN_EMAIL and');
    console.log('  ADMIN_PASSWORD before starting.');
    console.log('');
  }
});

// A stale dev server holding the port is the single most common way to be
// stopped on the first run, and Node's default output for it is a stack trace
// that says nothing useful.
server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  Port ${PORT} is already in use.\n`);
    console.error('  Another copy of this server is probably still running.');
    console.error('  Close it, or start on a different port:\n');
    console.error(`    Windows:  taskkill /F /IM node.exe`);
    console.error(`    or:       set PORT=5188 && npm run dev\n`);
    process.exit(1);
  }
  throw err;
});
