import fs from 'node:fs';
import path from 'node:path';
import { db } from './db.js';
import { saveTileFace, saveThumb, nano } from './storage.js';

async function run() {
  try {
    const tileImagePath = '/Users/niazig/Desktop/linxliving/testing-app/754048_-754054-capraia-white-splashback-gloss-_1.webp';
    if (!fs.existsSync(tileImagePath)) {
      console.error('Image not found:', tileImagePath);
      return;
    }
    
    const buffer = fs.readFileSync(tileImagePath);
    console.log('Read Capraia White Splashback image...');
    
    const faceUrl = await saveTileFace(buffer);
    const thumbUrl = await saveThumb(buffer, 'ptile');
    console.log('Saved to uploads:', faceUrl, thumbUrl);
    
    const id = `p_${nano()}`;
    const name = 'Capraia White Splashback';
    
    db.prepare(`
      INSERT INTO products (id, name, sku, category, material, finish, surfaces, sizes, faces, thumb, gloss, price, price_unit, pieces_per_box, coverage_sqm, description, sort)
      VALUES (@id, @name, @sku, @category, @material, @finish, @surfaces, @sizes, @faces, @thumb, @gloss, @price, @priceUnit, @piecesPerBox, @coverageSqm, @description, @sort)
    `).run({
      id,
      name,
      sku: 'CAPRAIA-WHT',
      category: 'marble',
      material: 'wall-panel',
      finish: 'gloss',
      surfaces: JSON.stringify(['wall']),
      sizes: JSON.stringify([{ w: 1200, h: 2400 }]), // Wall panels are typically large format
      faces: JSON.stringify([faceUrl]),
      thumb: thumbUrl,
      gloss: 0.85,
      price: 85.0,
      priceUnit: 'sqm',
      piecesPerBox: 2,
      coverageSqm: 5.76,
      description: 'Capraia White Gloss wall panel. Suitable for walls only.',
      sort: 2,
    });
    
    console.log('Successfully inserted Capraia White Splashback into database! ID:', id);
  } catch (err) {
    console.error(err);
  }
}

run();
