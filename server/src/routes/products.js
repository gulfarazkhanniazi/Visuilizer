import { Router } from 'express';
import { db, hydrateProduct } from '../db.js';
import {
  upload, uploadBulk, saveTileFace, saveThumb, removeUpload, nano,
} from '../storage.js';
import { materialModel, solidSwatch, parseHex } from '../materials.js';
import { requireAuth } from '../auth.js';

const router = Router();

router.get('/product-categories', (req, res) => {
  const rows = db.prepare(`
    SELECT c.id, c.name, c.sort, COUNT(p.id) AS count
      FROM product_categories c
      LEFT JOIN products p ON p.category = c.id AND p.active = 1
     GROUP BY c.id
     ORDER BY c.sort, c.name
  `).all();
  res.json(rows);
});

router.post('/product-categories', requireAuth, (req, res) => {
  const { id, name, sort = 0 } = req.body ?? {};
  if (!name) return res.status(400).json({ error: 'name is required' });
  const key = id || name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
  db.prepare(`
    INSERT INTO product_categories (id, name, sort) VALUES (?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET name = excluded.name, sort = excluded.sort
  `).run(key, name, sort);
  res.json({ id: key, name, sort });
});

router.get('/products', (req, res) => {
  const { category, surface, material, finish, q, sort = 'default', limit = 200, offset = 0 } = req.query;
  const where = ['active = 1'];
  // A non-numeric limit or offset falls back to the defaults rather than
  // failing the query.
  const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
  const params = { limit: num(limit, 200), offset: num(offset, 0) };

  if (category && category !== 'all') { where.push('category = @category'); params.category = category; }
  if (material) { where.push('material = @material'); params.material = material; }
  if (finish) { where.push('finish = @finish'); params.finish = finish; }
  // `surfaces` is a JSON array; a LIKE on the quoted key is exact enough here
  // because surface keys never contain each other as substrings.
  if (surface) { where.push("surfaces LIKE '%\"' || @surface || '\"%'"); params.surface = surface; }
  if (q) { where.push('(name LIKE @q OR sku LIKE @q)'); params.q = `%${q}%`; }

  // Whitelisted, because this lands straight in an ORDER BY clause.
  const ORDER = {
    default: 'sort, name',
    name: 'name COLLATE NOCASE',
    'name-desc': 'name COLLATE NOCASE DESC',
    newest: 'created_at DESC',
    'price-low': 'price IS NULL, price ASC',
    'price-high': 'price IS NULL, price DESC',
  };
  const rows = db.prepare(`
    SELECT * FROM products WHERE ${where.join(' AND ')}
    ORDER BY ${ORDER[sort] ?? ORDER.default} LIMIT @limit OFFSET @offset
  `).all(params);

  const total = db.prepare(
    `SELECT COUNT(*) AS n FROM products WHERE ${where.join(' AND ')}`,
  ).get(params).n;

  res.json({ total, items: rows.map(hydrateProduct) });
});

router.get('/products/:id', (req, res) => {
  const p = hydrateProduct(db.prepare('SELECT * FROM products WHERE id = ?').get(req.params.id));
  if (!p) return res.status(404).json({ error: 'Product not found' });
  res.json(p);
});

/**
 * Create a product. `faces` accepts several images -- a real tile range ships
 * multiple random faces so a laid floor does not visibly repeat.
 */
router.post('/products', requireAuth, upload.array('faces', 12), async (req, res, next) => {
  try {
    const b = req.body ?? {};
    if (!b.name) return res.status(400).json({ error: 'name is required' });

    const material = b.material ?? 'tile';
    const isSolid = materialModel(material) === 'solid';
    const color = isSolid ? parseHex(b.color) : (b.color ? parseHex(b.color) : null);

    const faces = [];
    for (const file of req.files ?? []) faces.push(await saveTileFace(file.buffer));
    if (!faces.length && b.faces) faces.push(...JSON.parse(b.faces));
    // Paint has no photograph to upload -- its swatch is generated from the
    // colour, which is also what the renderer actually uses.
    let swatch = null;
    if (!faces.length && isSolid) {
      swatch = await solidSwatch(color);
      faces.push(await saveTileFace(swatch, { size: 256 }));
    }
    if (!faces.length) return res.status(400).json({ error: 'At least one tile face is required' });

    const thumb = req.files?.[0]
      ? await saveThumb(req.files[0].buffer, 'ptile')
      : swatch
        ? await saveThumb(swatch, 'ptile')
        : faces[0];

    const id = b.id || `p_${nano()}`;
    db.prepare(`
      INSERT INTO products (id, name, sku, category, material, color, finish, surfaces, sizes,
                            faces, thumb, gloss, price, price_unit, pieces_per_box, coverage_sqm,
                            description, sort)
      VALUES (@id, @name, @sku, @category, @material, @color, @finish, @surfaces, @sizes,
              @faces, @thumb, @gloss, @price, @priceUnit, @piecesPerBox, @coverageSqm,
              @description, @sort)
    `).run({
      id,
      name: b.name,
      sku: b.sku ?? null,
      category: b.category ?? null,
      material,
      color,
      finish: b.finish ?? 'matt',
      surfaces: JSON.stringify(parseArr(b.surfaces, ['floor', 'wall'])),
      sizes: JSON.stringify(parseArr(b.sizes, [{ w: 600, h: 600 }])),
      faces: JSON.stringify(faces),
      thumb,
      gloss: Number(b.gloss ?? 0.25),
      price: b.price ? Number(b.price) : null,
      priceUnit: b.priceUnit ?? 'sqm',
      piecesPerBox: b.piecesPerBox ? Number(b.piecesPerBox) : null,
      coverageSqm: b.coverageSqm ? Number(b.coverageSqm) : null,
      description: b.description ?? null,
      sort: Number(b.sort ?? 0),
    });

    res.status(201).json(hydrateProduct(db.prepare('SELECT * FROM products WHERE id = ?').get(id)));
  } catch (e) { next(e); }
});

router.put('/products/:id', requireAuth, upload.array('faces', 12), async (req, res, next) => {
  try {
    const row = db.prepare('SELECT * FROM products WHERE id = ?').get(req.params.id);
    if (!row) return res.status(404).json({ error: 'Product not found' });
    const b = req.body ?? {};

    const material = b.material ?? row.material;
    const isSolid = materialModel(material) === 'solid';
    const color = b.color !== undefined && b.color !== ''
      ? parseHex(b.color)
      : (isSolid ? parseHex(row.color) : row.color);

    let faces = JSON.parse(row.faces);
    if (req.files?.length) {
      const added = [];
      for (const file of req.files) added.push(await saveTileFace(file.buffer));
      faces = b.replaceFaces === '1' ? added : [...faces, ...added];
    } else if (b.faces) {
      faces = parseArr(b.faces, faces);
    }

    // A paint product whose colour changed needs its swatch regenerating, or
    // the catalogue keeps showing the colour it used to be.
    let thumb = row.thumb;
    if (isSolid && color !== row.color && !req.files?.length) {
      const swatch = await solidSwatch(color);
      faces = [await saveTileFace(swatch, { size: 256 })];
      thumb = await saveThumb(swatch, 'ptile');
      for (const f of JSON.parse(row.faces)) await removeUpload(f);
      await removeUpload(row.thumb);
    }

    db.prepare(`
      UPDATE products SET name = @name, sku = @sku, category = @category, material = @material,
        color = @color, thumb = @thumb,
        finish = @finish, surfaces = @surfaces, sizes = @sizes, faces = @faces,
        gloss = @gloss, price = @price, price_unit = @priceUnit,
        pieces_per_box = @piecesPerBox, coverage_sqm = @coverageSqm,
        description = @description, active = @active, sort = @sort
      WHERE id = @id
    `).run({
      id: req.params.id,
      name: b.name ?? row.name,
      sku: b.sku ?? row.sku,
      category: b.category !== undefined ? b.category : row.category,
      material,
      color,
      thumb,
      finish: b.finish ?? row.finish,
      surfaces: JSON.stringify(parseArr(b.surfaces, JSON.parse(row.surfaces))),
      sizes: JSON.stringify(parseArr(b.sizes, JSON.parse(row.sizes))),
      faces: JSON.stringify(faces),
      gloss: b.gloss !== undefined ? Number(b.gloss) : row.gloss,
      price: b.price !== undefined && b.price !== '' ? Number(b.price) : row.price,
      priceUnit: b.priceUnit ?? row.price_unit ?? 'sqm',
      piecesPerBox: b.piecesPerBox !== undefined && b.piecesPerBox !== ''
        ? Number(b.piecesPerBox) : row.pieces_per_box,
      coverageSqm: b.coverageSqm !== undefined && b.coverageSqm !== ''
        ? Number(b.coverageSqm) : row.coverage_sqm,
      description: b.description !== undefined ? b.description : row.description,
      active: b.active !== undefined ? (b.active === '0' || b.active === false ? 0 : 1) : row.active,
      sort: b.sort !== undefined ? Number(b.sort) : row.sort,
    });

    res.json(hydrateProduct(db.prepare('SELECT * FROM products WHERE id = ?').get(req.params.id)));
  } catch (e) { next(e); }
});

router.delete('/products/:id', requireAuth, async (req, res, next) => {
  try {
    const row = db.prepare('SELECT * FROM products WHERE id = ?').get(req.params.id);
    if (!row) return res.status(404).json({ error: 'Product not found' });
    db.prepare('DELETE FROM products WHERE id = ?').run(req.params.id);
    for (const f of JSON.parse(row.faces)) await removeUpload(f);
    await removeUpload(row.thumb);
    res.json({ ok: true });
  } catch (e) { next(e); }
});

/**
 * Bulk upload: every image becomes its own single-face product, named after
 * the file. This is how a catalogue of a few hundred SKUs actually gets in.
 */
router.post('/products/bulk', requireAuth, uploadBulk.array('images', 200), async (req, res, next) => {
  try {
    const b = req.body ?? {};
    const created = [];
    for (const file of req.files ?? []) {
      const face = await saveTileFace(file.buffer);
      const thumb = await saveThumb(file.buffer, 'ptile');
      const id = `p_${nano()}`;
      const name = file.originalname.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ');
      db.prepare(`
        INSERT INTO products (id, name, sku, category, material, finish, surfaces, sizes, faces, thumb, gloss)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(id, name, null, b.category ?? null, b.material ?? 'tile', b.finish ?? 'matt',
             JSON.stringify(parseArr(b.surfaces, ['floor', 'wall'])),
             JSON.stringify(parseArr(b.sizes, [{ w: 600, h: 600 }])),
             JSON.stringify([face]), thumb, Number(b.gloss ?? 0.25));
      created.push(id);
    }
    res.status(201).json({ created: created.length, ids: created });
  } catch (e) { next(e); }
});

function parseArr(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  if (Array.isArray(value)) return value;
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : fallback;
  } catch {
    return String(value).split(',').map((s) => s.trim()).filter(Boolean);
  }
}

export default router;
