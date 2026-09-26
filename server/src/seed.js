/**
 * Seed the database with demo rooms and a starter tile catalogue.
 * Idempotent: re-running replaces the generated content but leaves anything
 * you uploaded through the admin panel alone.
 */
import sharp from 'sharp';
import path from 'node:path';
import { db, UPLOAD_DIR, hydrateRoom } from './db.js';
import { generateTileFace, buildRoom, roomSvg, surfaceMasks, buildPanorama } from './generate.js';

const ROOM_CATEGORIES = [
  { id: 'living-room', name: 'Living Room', sort: 1 },
  { id: 'bathroom',    name: 'Bathroom',    sort: 2 },
  { id: 'kitchen',     name: 'Kitchen',     sort: 3 },
  { id: 'bedroom',     name: 'Bedroom',     sort: 4 },
  { id: 'outdoor',     name: 'Outdoor',     sort: 5 },
];

const PRODUCT_CATEGORIES = [
  { id: 'marble',   name: 'Marble',        sort: 1 },
  { id: 'wood',     name: 'Wood & Plank',  sort: 2 },
  { id: 'concrete', name: 'Concrete',      sort: 3 },
  { id: 'terrazzo', name: 'Terrazzo',      sort: 4 },
  { id: 'glazed',   name: 'Glazed Ceramic',sort: 5 },
];

const PRODUCTS = [
  { id: 'p_carrara',   name: 'Carrara White',   kind: 'marble',   cat: 'marble',   finish: 'polished', gloss: 0.6,
    palette: [[236, 236, 233], [220, 220, 216], [150, 156, 162]], sizes: [{ w: 600, h: 600 }, { w: 800, h: 800 }, { w: 1200, h: 1200 }] },
  { id: 'p_calacatta', name: 'Calacatta Gold',  kind: 'marble',   cat: 'marble',   finish: 'polished', gloss: 0.72,
    palette: [[244, 241, 234], [232, 228, 218], [176, 146, 92]],  sizes: [{ w: 800, h: 800 }, { w: 1200, h: 2400 }] },
  { id: 'p_nero',      name: 'Nero Marquina',   kind: 'marble',   cat: 'marble',   finish: 'polished', gloss: 0.8,
    palette: [[28, 28, 30], [38, 38, 42], [226, 226, 224]],       sizes: [{ w: 600, h: 600 }, { w: 600, h: 1200 }] },
  { id: 'p_travert',   name: 'Roman Travertine',kind: 'marble',   cat: 'marble',   finish: 'honed',    gloss: 0.18,
    palette: [[218, 203, 178], [204, 187, 160], [176, 156, 126]], sizes: [{ w: 600, h: 600 }, { w: 600, h: 1200 }] },

  { id: 'p_oakn',      name: 'Natural Oak',     kind: 'wood',     cat: 'wood',     finish: 'matt',     gloss: 0.14,
    palette: [[196, 158, 112], [166, 126, 82], [150, 112, 70]],   sizes: [{ w: 200, h: 1200 }, { w: 150, h: 900 }] },
  { id: 'p_walnut',    name: 'Smoked Walnut',   kind: 'wood',     cat: 'wood',     finish: 'matt',     gloss: 0.12,
    palette: [[110, 78, 54], [78, 54, 38], [64, 44, 32]],         sizes: [{ w: 200, h: 1200 }, { w: 150, h: 900 }] },
  { id: 'p_ashg',      name: 'Grey Ash',        kind: 'wood',     cat: 'wood',     finish: 'matt',     gloss: 0.1,
    palette: [[176, 172, 166], [146, 142, 136], [128, 124, 118]], sizes: [{ w: 200, h: 1200 }, { w: 150, h: 900 }] },

  { id: 'p_cemgrey',   name: 'Cement Grey',     kind: 'concrete', cat: 'concrete', finish: 'matt',     gloss: 0.08,
    palette: [[164, 164, 162], [136, 136, 134], [120, 120, 118]], sizes: [{ w: 600, h: 600 }, { w: 600, h: 1200 }, { w: 1200, h: 1200 }] },
  { id: 'p_cemivory',  name: 'Cement Ivory',    kind: 'concrete', cat: 'concrete', finish: 'matt',     gloss: 0.08,
    palette: [[224, 218, 206], [204, 197, 184], [190, 182, 168]], sizes: [{ w: 600, h: 600 }, { w: 800, h: 800 }] },
  { id: 'p_cemanth',   name: 'Anthracite',      kind: 'concrete', cat: 'concrete', finish: 'matt',     gloss: 0.1,
    palette: [[68, 70, 74], [52, 54, 58], [44, 46, 50]],          sizes: [{ w: 600, h: 600 }, { w: 600, h: 1200 }] },

  { id: 'p_terrmono',  name: 'Terrazzo Mono',   kind: 'terrazzo', cat: 'terrazzo', finish: 'polished', gloss: 0.5,
    palette: [[232, 230, 224], [120, 122, 126], [190, 190, 186]], sizes: [{ w: 600, h: 600 }, { w: 800, h: 800 }] },
  { id: 'p_terrconf',  name: 'Terrazzo Confetti',kind: 'terrazzo',cat: 'terrazzo', finish: 'polished', gloss: 0.5,
    palette: [[240, 237, 230], [214, 122, 96], [96, 138, 130]],   sizes: [{ w: 600, h: 600 }, { w: 800, h: 800 }] },

  { id: 'p_subwhite',  name: 'Subway White',    kind: 'glazed',   cat: 'glazed',   finish: 'gloss',    gloss: 0.85,
    palette: [[248, 248, 246], [236, 236, 232], [255, 255, 255]], sizes: [{ w: 300, h: 100 }, { w: 150, h: 75 }],
    surfaces: ['wall', 'backsplash'] },
  { id: 'p_subsage',   name: 'Subway Sage',     kind: 'glazed',   cat: 'glazed',   finish: 'gloss',    gloss: 0.8,
    palette: [[186, 200, 182], [166, 182, 162], [204, 216, 200]], sizes: [{ w: 300, h: 100 }, { w: 150, h: 75 }],
    surfaces: ['wall', 'backsplash'] },
  { id: 'p_subnavy',   name: 'Subway Navy',     kind: 'glazed',   cat: 'glazed',   finish: 'gloss',    gloss: 0.85,
    palette: [[42, 58, 92], [32, 46, 76], [70, 90, 128]],         sizes: [{ w: 300, h: 100 }, { w: 150, h: 75 }],
    surfaces: ['wall', 'backsplash'] },
  { id: 'p_terracot',  name: 'Terracotta',      kind: 'glazed',   cat: 'glazed',   finish: 'matt',     gloss: 0.2,
    palette: [[196, 118, 82], [176, 100, 68], [210, 138, 104]],   sizes: [{ w: 200, h: 200 }, { w: 150, h: 150 }] },
];

const PRICES = { marble: 62, wood: 44, concrete: 31, terrazzo: 55, glazed: 26 };

const ROOMS = [
  {
    id: 'room_living', name: 'Modern Living Room', category: 'living-room',
    spec: {
      width: 1600, height: 1100, f: 1000, camY: 1.4, halfW: 1.9, roomH: 2.7, backZ: -5.4,
      window: { z0: -4.6, z1: -3.1, y0: 0.85, y1: 2.25, side: 'right' },
      palette: {
        wallTop: '#efeae2', wallBottom: '#e2dcd2', wallSide: '#d8d2c8', wallSideLit: '#f2ede5',
        ceiling: '#f6f3ee', floorFar: '#b9ae9e', floorNear: '#cbc1b2', trim: '#ffffff', prop: '#8d8375',
      },
      props: [
        { x: -0.55, z: -3.55, w: 2.1, d: 0.9, h: 0.001, type: 'rug', color: '#b9b0a2' },
        { x: -1.0,  z: -4.6,  w: 1.9, d: 0.85, h: 0.78, color: '#6f7b86' },
        { x: 1.15,  z: -4.5,  w: 0.55, d: 0.55, h: 0.52, color: '#8a7f70' },
      ],
    },
  },
  {
    id: 'room_bath', name: 'Contemporary Bathroom', category: 'bathroom',
    spec: {
      width: 1600, height: 1100, f: 1050, camY: 1.45, halfW: 1.35, roomH: 2.55, backZ: -4.2,
      window: { z0: -3.7, z1: -2.9, y0: 1.15, y1: 2.15, side: 'left' },
      palette: {
        wallTop: '#eef1f3', wallBottom: '#dfe4e8', wallSide: '#d4dade', wallSideLit: '#f0f3f5',
        ceiling: '#f7f9fa', floorFar: '#b4bcc2', floorNear: '#c8cfd4', trim: '#ffffff', prop: '#f2f4f5',
      },
      props: [
        { x: 0.75, z: -3.9, w: 1.0, d: 0.5, h: 0.85, color: '#f4f6f7' },
        { x: -0.85, z: -3.95, w: 0.85, d: 0.7, h: 0.55, color: '#eceff1' },
      ],
    },
  },
  {
    id: 'room_kitchen', name: 'Open Kitchen', category: 'kitchen',
    spec: {
      width: 1600, height: 1100, f: 980, camY: 1.5, halfW: 1.8, roomH: 2.65, backZ: -5.0,
      window: { z0: -4.4, z1: -3.4, y0: 1.05, y1: 2.1, side: 'right' },
      palette: {
        wallTop: '#f3f1ec', wallBottom: '#e8e5de', wallSide: '#ded9d1', wallSideLit: '#f6f4ef',
        ceiling: '#faf8f5', floorFar: '#b0a89c', floorNear: '#c4bcb0', trim: '#ffffff', prop: '#3f4c52',
      },
      props: [
        { x: -1.05, z: -4.55, w: 1.5, d: 0.65, h: 0.92, color: '#39464c' },
        { x: 0.95,  z: -4.55, w: 1.3, d: 0.65, h: 0.92, color: '#39464c' },
        { x: 0.0,   z: -3.5,  w: 1.6, d: 0.85, h: 0.9,  color: '#5d6a70' },
      ],
    },
  },
  {
    id: 'room_bedroom', name: 'Calm Bedroom', category: 'bedroom',
    spec: {
      width: 1600, height: 1100, f: 1020, camY: 1.35, halfW: 1.95, roomH: 2.7, backZ: -5.6,
      window: { z0: -4.9, z1: -3.5, y0: 0.9, y1: 2.2, side: 'left' },
      palette: {
        wallTop: '#f0ece6', wallBottom: '#e4dfd7', wallSide: '#dad4cb', wallSideLit: '#f4f0ea',
        ceiling: '#f8f6f2', floorFar: '#b7ada0', floorNear: '#cac1b5', trim: '#ffffff', prop: '#9a9186',
      },
      props: [
        { x: 0.1,  z: -4.75, w: 2.0, d: 1.9, h: 0.62, color: '#b9b2a8' },
        { x: -1.3, z: -4.9,  w: 0.5, d: 0.45, h: 0.55, color: '#8f857a' },
        { x: 0.45, z: -3.2,  w: 2.4, d: 1.1, h: 0.001, type: 'rug', color: '#c2b9ac' },
      ],
    },
  },
  {
    id: 'room_terrace', name: 'Garden Terrace', category: 'outdoor',
    spec: {
      width: 1600, height: 1100, f: 940, camY: 1.55, halfW: 2.3, roomH: 2.9, backZ: -6.2,
      window: { z0: -5.6, z1: -3.8, y0: 0.6, y1: 2.4, side: 'right' },
      palette: {
        wallTop: '#e9e6df', wallBottom: '#dcd8cf', wallSide: '#cfcabf', wallSideLit: '#eeebe4',
        ceiling: '#dfe7ee', floorFar: '#a9a498', floorNear: '#bfb9ac', trim: '#ffffff', prop: '#7d8a72',
      },
      props: [
        { x: -1.2, z: -5.3, w: 1.0, d: 0.9, h: 0.75, color: '#6f7d64' },
        { x: 1.35, z: -5.2, w: 0.7, d: 0.7, h: 1.05, color: '#7f8b6e' },
      ],
    },
  },
];


const PANORAMAS = [
  {
    id: 'pano_loft', name: 'Loft 360', category: 'living-room',
    spec: {
      width: 2560, height: 1280, halfW: 2.4, halfD: 3.0, roomH: 2.8, camY: 1.55,
      window: { wall: 'right', u0: -1.1, u1: 1.1, v0: 0.85, v1: 2.3 },
      palette: { floor: [176, 166, 152], wall: [232, 228, 220], ceiling: [246, 244, 240] },
      seed: 11,
    },
  },
  {
    id: 'pano_bath', name: 'Spa Bathroom 360', category: 'bathroom',
    spec: {
      width: 2560, height: 1280, halfW: 1.7, halfD: 2.1, roomH: 2.6, camY: 1.5,
      window: { wall: 'left', u0: -0.7, u1: 0.7, v0: 1.1, v1: 2.1 },
      palette: { floor: [172, 180, 186], wall: [226, 232, 236], ceiling: [244, 248, 250] },
      seed: 23,
    },
  },
];

async function seedPanoramas() {
  const insert = db.prepare(`
    INSERT INTO rooms (id, name, category, image, thumb, width, height, data, kind, is_custom, sort)
    VALUES (@id, @name, @category, @image, @thumb, @width, @height, @data, '360', 0, @sort)
    ON CONFLICT(id) DO UPDATE SET
      name = excluded.name, category = excluded.category, image = excluded.image,
      thumb = excluded.thumb, width = excluded.width, height = excluded.height,
      data = excluded.data, kind = excluded.kind
  `);

  let sort = 100;
  for (const p of PANORAMAS) {
    const pano = buildPanorama(p.spec);
    const file = `${p.id}.jpg`;
    const thumbFile = `${p.id}-thumb.jpg`;

    await sharp(pano.data, { raw: pano.info })
      .jpeg({ quality: 90, mozjpeg: true })
      .toFile(path.join(UPLOAD_DIR, file));
    // Thumbnail from the middle of the strip, which reads as a room rather
    // than the stretched poles an equirect crop would give.
    await sharp(pano.data, { raw: pano.info })
      .extract({
        left: Math.round(pano.info.width * 0.32),
        top: Math.round(pano.info.height * 0.3),
        width: Math.round(pano.info.width * 0.28),
        height: Math.round(pano.info.height * 0.4),
      })
      .resize(480, 320, { fit: 'cover' })
      .jpeg({ quality: 80, mozjpeg: true })
      .toFile(path.join(UPLOAD_DIR, thumbFile));

    insert.run({
      id: p.id,
      name: p.name,
      category: p.category,
      image: `/uploads/${file}`,
      thumb: `/uploads/${thumbFile}`,
      width: pano.info.width,
      height: pano.info.height,
      data: JSON.stringify({
        objectList: pano.surfaces.map((s, i) => ({ ...s, order: i })),
        settings: { blurRadius: 9, camY: p.spec.camY },
      }),
      sort: sort++,
    });
    process.stdout.write(`  360   ${p.name}  (${pano.surfaces.length} surfaces)
`);
  }
}

// ------------------------------------------------------------------ run ----

async function seedProducts() {
  const insert = db.prepare(`
    INSERT INTO products (id, name, sku, category, material, finish, surfaces, sizes, faces,
                          thumb, gloss, price, price_unit, pieces_per_box, coverage_sqm,
                          description, sort)
    VALUES (@id, @name, @sku, @category, @material, @finish, @surfaces, @sizes, @faces,
            @thumb, @gloss, @price, @priceUnit, @piecesPerBox, @coverageSqm,
            @description, @sort)
    ON CONFLICT(id) DO UPDATE SET
      name = excluded.name, category = excluded.category, material = excluded.material,
      finish = excluded.finish, surfaces = excluded.surfaces, sizes = excluded.sizes,
      faces = excluded.faces, thumb = excluded.thumb, gloss = excluded.gloss,
      price = excluded.price, price_unit = excluded.price_unit,
      pieces_per_box = excluded.pieces_per_box, coverage_sqm = excluded.coverage_sqm,
      description = excluded.description
  `);

  let sort = 0;
  for (const p of PRODUCTS) {
    // A box holds roughly 1.4 m2 of tile, rounded to a whole number of pieces --
    // close enough to how real ranges are packed for the calculator to be useful.
    const [s0] = p.sizes;
    const pieceSqm = (s0.w / 1000) * (s0.h / 1000);
    p.piecesPerBox = Math.max(1, Math.round(1.4 / pieceSqm));
    p.coverageSqm = Math.round(p.piecesPerBox * pieceSqm * 1000) / 1000;
    p.price = p.price ?? PRICES[p.kind] ?? 28;
    p.description = p.description
      ?? `${p.name} - ${p.finish} finish ${p.kind === 'wood' ? 'plank' : 'tile'}.`;

    const faces = [];
    // Four random faces per product: enough that a laid floor stops repeating.
    for (let i = 0; i < 4; i++) {
      const { data, info } = generateTileFace(p.kind, p.palette, hashSeed(p.id) + i * 5081);
      const file = `tile-${p.id}-${i}.jpg`;
      await sharp(data, { raw: info }).jpeg({ quality: 92, mozjpeg: true })
        .toFile(path.join(UPLOAD_DIR, file));
      faces.push(`/uploads/${file}`);
    }

    const thumbFile = `tile-${p.id}-thumb.jpg`;
    const { data, info } = generateTileFace(p.kind, p.palette, hashSeed(p.id), 320);
    await sharp(data, { raw: info }).jpeg({ quality: 84, mozjpeg: true })
      .toFile(path.join(UPLOAD_DIR, thumbFile));

    insert.run({
      id: p.id,
      name: p.name,
      sku: p.id.replace('p_', '').toUpperCase(),
      category: p.cat,
      material: p.kind === 'wood' ? 'wood' : p.kind === 'marble' ? 'marble' : 'tile',
      finish: p.finish,
      surfaces: JSON.stringify(p.surfaces ?? ['floor', 'wall', 'backsplash', 'countertop', 'outdoor']),
      sizes: JSON.stringify(p.sizes),
      faces: JSON.stringify(faces),
      thumb: `/uploads/${thumbFile}`,
      gloss: p.gloss,
      price: p.price,
      priceUnit: 'sqm',
      piecesPerBox: p.piecesPerBox,
      coverageSqm: p.coverageSqm,
      description: p.description,
      sort: sort++,
    });
    process.stdout.write(`  tile  ${p.name}\n`);
  }
}

async function seedRooms() {
  const insert = db.prepare(`
    INSERT INTO rooms (id, name, category, image, thumb, width, height, data, is_custom, sort)
    VALUES (@id, @name, @category, @image, @thumb, @width, @height, @data, 0, @sort)
    ON CONFLICT(id) DO UPDATE SET
      name = excluded.name, category = excluded.category, image = excluded.image,
      thumb = excluded.thumb, width = excluded.width, height = excluded.height,
      data = excluded.data
  `);

  let sort = 0;
  for (const r of ROOMS) {
    const room = buildRoom(r.spec);
    const svg = roomSvg(room);
    const masks = surfaceMasks(room);

    const file = `${r.id}.jpg`;
    const thumbFile = `${r.id}-thumb.jpg`;

    // A touch of blur and grain moves the render away from "flat vector art"
    // and gives the lighting plate some real high-frequency content to work on.
    const base = sharp(Buffer.from(svg)).resize(room.width, room.height);
    const rendered = await base.jpeg({ quality: 95, mozjpeg: true }).toBuffer();
    await sharp(rendered).blur(0.7).modulate({ saturation: 0.96 })
      .jpeg({ quality: 92, mozjpeg: true })
      .toFile(path.join(UPLOAD_DIR, file));
    await sharp(rendered).resize(480, 320, { fit: 'cover' })
      .jpeg({ quality: 80, mozjpeg: true })
      .toFile(path.join(UPLOAD_DIR, thumbFile));

    const objectList = room.surfaces.map((s, i) => ({
      name: s.name,
      label: s.label,
      product_surface: s.product_surface,
      quad: s.quad,
      realSize: s.realSize,
      mask: masks[s.name],
      isMain: s.name === 'floor',
      order: i,
      defaults: s.product_surface === 'floor'
        ? { tileSize: { w: 600, h: 600 }, layout: 'grid', grout: { size: 2, color: '#c9c9c4' } }
        : { tileSize: { w: 300, h: 600 }, layout: 'brick', grout: { size: 2, color: '#f5f5f2' } },
    }));

    insert.run({
      id: r.id,
      name: r.name,
      category: r.category,
      image: `/uploads/${file}`,
      thumb: `/uploads/${thumbFile}`,
      width: room.width,
      height: room.height,
      data: JSON.stringify({ objectList, settings: { blurRadius: 7 } }),
      sort: sort++,
    });
    process.stdout.write(`  room  ${r.name}  (${objectList.length} surfaces)\n`);
  }
}

function hashSeed(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

async function main() {
  const cat = db.prepare('INSERT INTO room_categories (id, name, sort) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name');
  for (const c of ROOM_CATEGORIES) cat.run(c.id, c.name, c.sort);
  const pcat = db.prepare('INSERT INTO product_categories (id, name, sort) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name');
  for (const c of PRODUCT_CATEGORIES) pcat.run(c.id, c.name, c.sort);

  console.log('Seeding tile catalogue...');
  await seedProducts();
  console.log('Seeding demo rooms...');
  await seedRooms();
  console.log('Seeding 360 rooms...');
  await seedPanoramas();

  const rooms = db.prepare('SELECT * FROM rooms').all().map(hydrateRoom);
  console.log(`\nDone: ${rooms.length} rooms, ${PRODUCTS.length} products.`);
}

main().catch((e) => { console.error(e); process.exit(1); });
