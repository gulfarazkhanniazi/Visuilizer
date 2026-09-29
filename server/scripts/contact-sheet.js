/** Tile every PNG in a folder into one contact sheet: node scripts/contact-sheet.js <dir> <out.png> [cols] [tileW] */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
const [dir, out, colsArg = '3', tileArg = '600'] = process.argv.slice(2);
const cols = Number(colsArg); const tw = Number(tileArg); const th = Math.round(tw * 2 / 3);
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.png')).sort();
const tiles = await Promise.all(files.map((f) => sharp(path.join(dir, f)).resize(tw, th, { fit: 'contain', background: '#222' }).png().toBuffer()));
const rows = Math.ceil(tiles.length / cols);
await sharp({ create: { width: cols * tw, height: rows * th, channels: 3, background: '#222' } })
  .composite(tiles.map((b, i) => ({ input: b, left: (i % cols) * tw, top: Math.floor(i / cols) * th }))).png().toFile(out);
console.log(files.join('\n'));
