/**
 * Import first in any test that loads server modules: points DATA_DIR at a
 * throwaway directory (sharing the cached models) so tests never open, and
 * never checkpoint, the real database. ES modules evaluate imports in order,
 * so this runs before db.js is loaded.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (!process.env.DATA_DIR) {
  const real = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'data');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'scanner-test-'));
  fs.symlinkSync(path.join(real, 'models'), path.join(dir, 'models'));
  process.env.DATA_DIR = dir;
  process.on('exit', () => fs.rmSync(dir, { recursive: true, force: true }));
}
