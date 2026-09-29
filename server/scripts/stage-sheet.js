/** Grid of chosen debug stages for every scan in a report folder:
 *  node scripts/stage-sheet.js <root> <out.png> 07-plane-detection.png 06-point-cloud.png ... */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
const [root, out, ...names] = process.argv.slice(2);
const dirs = fs.readdirSync(root).filter((d) => fs.statSync(path.join(root, d)).isDirectory()).sort();
const tw = 360; const th = 250;
const comp = [];
for (const [r, d] of dirs.entries()) {
  for (const [c, n] of names.entries()) {
    const f = path.join(root, d, n);
    if (!fs.existsSync(f)) continue;
    comp.push({ input: await sharp(f).resize(tw, th, { fit: 'contain', background: '#000' }).png().toBuffer(), left: c * tw, top: r * th });
  }
}
await sharp({ create: { width: tw * names.length, height: th * dirs.length, channels: 3, background: '#000' } }).composite(comp).png().toFile(out);
