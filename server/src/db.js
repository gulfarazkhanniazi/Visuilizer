import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const DATA_DIR = path.resolve(__dirname, '..', 'data');
export const UPLOAD_DIR = path.join(DATA_DIR, 'uploads');

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

export const db = new Database(path.join(DATA_DIR, 'visualizer.db'));
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
CREATE TABLE IF NOT EXISTS vendor (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  name          TEXT NOT NULL,
  logo          TEXT,
  primary_color TEXT DEFAULT '#2563eb',
  settings      TEXT NOT NULL DEFAULT '{}',
  updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS room_categories (
  id    TEXT PRIMARY KEY,
  name  TEXT NOT NULL,
  sort  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rooms (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  category    TEXT REFERENCES room_categories(id) ON DELETE SET NULL,
  image       TEXT NOT NULL,
  thumb       TEXT,
  width       INTEGER NOT NULL,
  height      INTEGER NOT NULL,
  data        TEXT NOT NULL,          -- objectList + settings, as JSON
  is_custom   INTEGER NOT NULL DEFAULT 0,
  owner       TEXT,                   -- visitor id, for user-uploaded rooms
  sort        INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rooms_category ON rooms(category);
CREATE INDEX IF NOT EXISTS idx_rooms_owner ON rooms(owner);

CREATE TABLE IF NOT EXISTS product_categories (
  id    TEXT PRIMARY KEY,
  name  TEXT NOT NULL,
  sort  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS products (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  sku         TEXT,
  category    TEXT REFERENCES product_categories(id) ON DELETE SET NULL,
  material    TEXT NOT NULL DEFAULT 'tile',
  finish      TEXT NOT NULL DEFAULT 'matt',
  surfaces    TEXT NOT NULL DEFAULT '["floor","wall"]',
  sizes       TEXT NOT NULL DEFAULT '[]',   -- [{w,h}] in mm
  faces       TEXT NOT NULL DEFAULT '[]',   -- random-face image urls
  thumb       TEXT,
  gloss       REAL NOT NULL DEFAULT 0.25,
  price       REAL,
  active      INTEGER NOT NULL DEFAULT 1,
  sort        INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

CREATE TABLE IF NOT EXISTS shares (
  code       TEXT PRIMARY KEY,
  payload    TEXT NOT NULL,
  preview    TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS leads (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT,
  email      TEXT,
  phone      TEXT,
  message    TEXT,
  context    TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- A visitor's saved looks: the full per-surface state plus a preview image,
-- so they can come back to a scheme instead of rebuilding it.
CREATE TABLE IF NOT EXISTS saved_rooms (
  id         TEXT PRIMARY KEY,
  owner      TEXT NOT NULL,
  room_id    TEXT NOT NULL,
  name       TEXT NOT NULL,
  payload    TEXT NOT NULL,
  preview    TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_saved_owner ON saved_rooms(owner);

CREATE TABLE IF NOT EXISTS wishlist (
  owner      TEXT NOT NULL,
  product_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (owner, product_id)
);

CREATE TABLE IF NOT EXISTS stores (
  id      TEXT PRIMARY KEY,
  name    TEXT NOT NULL,
  address TEXT,
  city    TEXT,
  phone   TEXT,
  email   TEXT,
  lat     REAL,
  lng     REAL,
  sort    INTEGER NOT NULL DEFAULT 0
);
`);

/**
 * Add a column to an existing table if it is not already there.
 * The schema above only runs CREATE TABLE IF NOT EXISTS, so databases created
 * by an earlier version need their new columns filling in explicitly.
 */
function addColumn(table, column, definition) {
  const cols = db.prepare(`PRAGMA table_info(${table})`).all();
  if (cols.some((c) => c.name === column)) return;
  db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
}

addColumn('products', 'price_unit', "TEXT NOT NULL DEFAULT 'sqm'");
addColumn('products', 'pieces_per_box', 'INTEGER');
addColumn('products', 'coverage_sqm', 'REAL');
addColumn('products', 'description', 'TEXT');
// '2d' | '360' | '3d' -- which renderer a room needs.
addColumn('rooms', 'kind', "TEXT NOT NULL DEFAULT '2d'");
// The flat colour a `solid` product (paint) is rendered as. Null for every
// material that has a photographed face instead.
addColumn('products', 'color', 'TEXT');
// The glTF/GLB file a '3d' room is rendered from. Null for photos and panoramas.
addColumn('rooms', 'model', 'TEXT');

const hasVendor = db.prepare('SELECT COUNT(*) AS n FROM vendor').get().n;
if (!hasVendor) {
  db.prepare(
    'INSERT INTO vendor (id, name, primary_color, settings) VALUES (1, ?, ?, ?)',
  ).run(
    'Surface Studio',
    '#2563eb',
    JSON.stringify({
      watermark: 'Surface Studio',
      allowUpload: true,
      allowDownload: true,
      allowCompare: true,
      allowShare: true,
      defaultBlurRadius: 6,
      contactEmail: '',
      contactPhone: '',
      currency: 'USD',
      currencySymbol: '$',
      allowInquiry: true,
      allowCalculator: true,
      allowWishlist: true,
      allowSaveRoom: true,
      allowPdf: true,
      allowPrice: true,
      wastagePercent: 10,
    }),
  );
}

/** Parse the JSON columns so routes deal in plain objects. */
export function hydrateRoom(row) {
  if (!row) return null;
  const data = JSON.parse(row.data);
  return {
    id: row.id,
    name: row.name,
    category: row.category,
    image: row.image,
    thumb: row.thumb || row.image,
    width: row.width,
    height: row.height,
    kind: row.kind ?? '2d',
    model: row.model ?? null,
    isCustom: !!row.is_custom,
    objectList: data.objectList ?? [],
    settings: data.settings ?? {},
    createdAt: row.created_at,
  };
}

export function hydrateProduct(row) {
  if (!row) return null;
  const faces = JSON.parse(row.faces);
  return {
    id: row.id,
    name: row.name,
    sku: row.sku,
    category: row.category,
    material: row.material,
    finish: row.finish,
    surfaces: JSON.parse(row.surfaces),
    sizes: JSON.parse(row.sizes),
    faces,
    image: faces[0] ?? row.thumb,
    thumb: row.thumb ?? faces[0],
    gloss: row.gloss,
    color: row.color,
    price: row.price,
    priceUnit: row.price_unit ?? 'sqm',
    piecesPerBox: row.pieces_per_box,
    coverageSqm: row.coverage_sqm,
    description: row.description,
    active: !!row.active,
  };
}
