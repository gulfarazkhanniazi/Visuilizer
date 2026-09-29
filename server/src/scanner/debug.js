/**
 * Debug images for one scan, so that when a wall comes out wrong it is
 * possible to see WHICH stage went wrong rather than only the final result.
 *
 * Written only when a scan is run with `debug` (or SCAN_DEBUG=1); nothing here
 * is on the production path otherwise.
 */
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

export const PALETTE = [
  [230, 25, 75], [60, 180, 75], [67, 99, 216], [245, 130, 49], [145, 30, 180],
  [66, 212, 244], [240, 50, 230], [191, 239, 69], [250, 190, 212], [70, 153, 144],
  [220, 190, 255], [154, 99, 36],
];
const CLASS_COLORS = {
  0: [40, 40, 40], 1: [120, 170, 255], 2: [240, 200, 60], 3: [200, 200, 200], 4: [255, 80, 80], 5: [255, 140, 0],
};

/** Turbo colormap, t in 0..1. */
function turbo(t) {
  const x = Math.min(1, Math.max(0, t));
  const r = 34.61 + x * (1172.33 - x * (10793.56 - x * (33300.12 - x * (38394.49 - x * 14825.05))));
  const g = 23.31 + x * (557.33 + x * (1225.33 - x * (3574.96 - x * (1073.77 + x * 707.56))));
  const b = 27.2 + x * (3211.1 - x * (15327.97 - x * (27814 - x * (22569.18 - x * 6838.66))));
  return [r, g, b].map((v) => Math.max(0, Math.min(255, Math.round(v))));
}

export class DebugWriter {
  constructor(dir, { photo, w, h }) {
    this.dir = dir;
    this.photo = photo;
    this.w = w;
    this.h = h;
    this.files = [];
    this.pending = [];
    fs.mkdirSync(dir, { recursive: true });
  }

  /** Photo at a given size as raw RGB, cached. */
  async base(W, H) {
    this.bases ??= new Map();
    const key = `${W}x${H}`;
    if (!this.bases.has(key)) {
      this.bases.set(key, await sharp(this.photo).removeAlpha().resize(W, H, { fit: 'fill' }).raw().toBuffer());
    }
    return this.bases.get(key);
  }

  save(name, rgb, W, H, { svg = null, outW = null } = {}) {
    const file = path.join(this.dir, name);
    this.files.push(name);
    const target = outW ?? Math.min(960, Math.max(W, 640));
    let img = sharp(Buffer.from(rgb), { raw: { width: W, height: H, channels: 3 } })
      .resize(target, Math.round((H / W) * target), { kernel: W < target ? 'nearest' : 'lanczos3' });
    if (svg) {
      const tw = target; const th = Math.round((H / W) * target);
      img = sharp(Buffer.from(rgb), { raw: { width: W, height: H, channels: 3 } })
        .resize(tw, th, { kernel: W < tw ? 'nearest' : 'lanczos3' })
        .composite([{ input: Buffer.from(svg(tw / W, tw, th)) }]);
    }
    this.pending.push(img.png().toFile(file));
  }

  async original() {
    const W = Math.min(960, this.w); const H = Math.round((this.h / this.w) * W);
    this.save('01-original.png', await this.base(W, H), W, H);
  }

  async classes(grid) {
    const { GW, GH, cls } = grid;
    const photo = await this.base(GW, GH);
    const rgb = new Uint8Array(GW * GH * 3);
    for (let i = 0; i < GW * GH; i++) {
      const c = CLASS_COLORS[cls[i]] ?? [0, 0, 0];
      for (let k = 0; k < 3; k++) rgb[i * 3 + k] = Math.round(photo[i * 3 + k] * 0.35 + c[k] * 0.65);
    }
    this.save('02-segmentation.png', rgb, GW, GH);
  }

  depth(grid) {
    const { GW, GH, Z } = grid;
    const inv = Array.from(Z, (z) => 1 / Math.max(z, 1e-3)).sort((a, b) => a - b);
    const lo = inv[Math.floor(inv.length * 0.02)]; const hi = inv[Math.floor(inv.length * 0.98)];
    const rgb = new Uint8Array(GW * GH * 3);
    for (let i = 0; i < GW * GH; i++) {
      const c = turbo((1 / Math.max(Z[i], 1e-3) - lo) / (hi - lo || 1));
      rgb.set(c, i * 3);
    }
    this.save('03-depth.png', rgb, GW, GH);
  }

  edges(grid, edges) {
    const { GW, GH } = grid;
    const rgb = new Uint8Array(GW * GH * 3);
    for (let i = 0; i < GW * GH; i++) {
      const j = Math.min(1, edges.jump[i] / 0.15);
      const c = Math.min(1, edges.crease[i] / 0.02);
      rgb[i * 3] = Math.round(255 * j);
      rgb[i * 3 + 1] = Math.round(255 * c);
      rgb[i * 3 + 2] = Math.round(60 * (1 - j));
    }
    this.save('04-depth-boundaries.png', rgb, GW, GH);
  }

  normals(grid, W) {
    const { GW, GH } = grid;
    const rgb = new Uint8Array(GW * GH * 3);
    for (let i = 0; i < GW * GH; i++) {
      if (!W.ok[i]) continue;
      rgb[i * 3] = Math.round((W.nX[i] * 0.5 + 0.5) * 255);
      rgb[i * 3 + 1] = Math.round((W.nY[i] * 0.5 + 0.5) * 255);
      rgb[i * 3 + 2] = Math.round((-W.nZ[i] * 0.5 + 0.5) * 255);
    }
    this.save('05-normal-map.png', rgb, GW, GH);
  }

  /** Top-down (bird's-eye) view of the wall points, coloured by wall. */
  pointCloud(grid, W, label, walls, { name = '06-point-cloud.png', corners = [], extents = null } = {}) {
    const S = 640;
    const pts = [];
    for (let i = 0; i < grid.cls.length; i += 1) {
      if (!W.valid[i] || (grid.cls[i] !== 1 && grid.cls[i] !== 4 && grid.cls[i] !== 5)) continue;
      pts.push(i);
    }
    const zs = pts.map((i) => W.Z[i]).sort((a, b) => a - b);
    const xs = pts.map((i) => W.X[i]).sort((a, b) => a - b);
    const zMax = Math.max(2, zs[Math.floor(zs.length * 0.99)] ?? 5) * 1.1;
    const xAbs = Math.max(1.5, Math.abs(xs[Math.floor(xs.length * 0.01)] ?? 3), Math.abs(xs[Math.floor(xs.length * 0.99)] ?? 3)) * 1.1;
    const span = Math.max(zMax, 2 * xAbs);
    const px = (X) => S / 2 + (X / span) * S;
    const pz = (Z) => S - 20 - (Z / span) * (S - 40);
    const rgb = new Uint8Array(S * S * 3).fill(18);
    for (const i of pts) {
      const x = Math.round(px(W.X[i])); const y = Math.round(pz(W.Z[i]));
      if (x < 0 || y < 0 || x >= S || y >= S) continue;
      const l = label[i];
      const c = l >= 0 ? PALETTE[l % PALETTE.length] : grid.cls[i] === 1 ? [110, 110, 110] : [90, 50, 20];
      rgb.set(c, (y * S + x) * 3);
    }
    const svg = () => {
      const parts = [`<circle cx="${px(0)}" cy="${pz(0)}" r="6" fill="#fff"/>`,
        `<text x="${px(0) + 8}" y="${pz(0) - 4}" fill="#fff" font-size="12" font-family="sans-serif">camera</text>`];
      walls.forEach((wl, k) => {
        const e = extents?.[k];
        if (!e) return;
        const c = PALETTE[k % PALETTE.length];
        const a = e.p0; const b = e.p1;
        parts.push(`<line x1="${px(a[0])}" y1="${pz(a[1])}" x2="${px(b[0])}" y2="${pz(b[1])}" stroke="rgb(${c})" stroke-width="3"/>`);
        parts.push(`<text x="${px((a[0] + b[0]) / 2)}" y="${pz((a[1] + b[1]) / 2) - 6}" fill="rgb(${c})" font-size="13" font-family="sans-serif">${e.id}</text>`);
      });
      for (const c of corners) {
        const col = c.inferred ? '#ff00ff' : '#00ff66';
        parts.push(`<circle cx="${px(c.xz[0])}" cy="${pz(c.xz[1])}" r="6" fill="${col}" stroke="#000"/>`);
      }
      parts.push(`<text x="10" y="18" fill="#aaa" font-size="12" font-family="sans-serif">top-down, ${span.toFixed(1)} m across</text>`);
      return `<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}">${parts.join('')}</svg>`;
    };
    this.save(name, rgb, S, S, { svg, outW: S });
  }

  async labels(grid, label, name, { lines = [], scale = 1 } = {}) {
    const { GW, GH } = grid;
    const photo = await this.base(GW, GH);
    const rgb = new Uint8Array(GW * GH * 3);
    for (let i = 0; i < GW * GH; i++) {
      const l = label[i];
      const c = l >= 0 ? PALETTE[l % PALETTE.length] : null;
      for (let k = 0; k < 3; k++) rgb[i * 3 + k] = c ? Math.round(photo[i * 3 + k] * 0.35 + c[k] * 0.65) : Math.round(photo[i * 3 + k] * 0.5);
    }
    const svg = lines.length ? (s, tw, th) => {
      const parts = lines.map((L) => `<line x1="${L[0] * s * scale}" y1="${L[1] * s * scale}" x2="${L[2] * s * scale}" y2="${L[3] * s * scale}" stroke="${L[4] ?? '#ffff00'}" stroke-width="${L[5] ?? 1.5}"/>`);
      return `<svg xmlns="http://www.w3.org/2000/svg" width="${tw}" height="${th}">${parts.join('')}</svg>`;
    } : null;
    this.save(name, rgb, GW, GH, { svg });
  }

  /** Photo with an SVG overlay drawn in full-image pixel coordinates. */
  async overlay(name, draw) {
    const W = Math.min(960, this.w); const H = Math.round((this.h / this.w) * W);
    const s0 = W / this.w;
    this.save(name, await this.base(W, H), W, H, {
      svg: (s, tw, th) => `<svg xmlns="http://www.w3.org/2000/svg" width="${tw}" height="${th}">${draw(s0 * s)}</svg>`,
    });
  }

  /** Raster masks at refinement resolution, one colour per wall. */
  async masks(name, masks, RW, RH, { dim = null } = {}) {
    const photo = await this.base(RW, RH);
    const rgb = new Uint8Array(RW * RH * 3);
    for (let i = 0; i < RW * RH; i++) {
      let c = null;
      for (let k = 0; k < masks.length; k++) if (masks[k][i]) { c = PALETTE[k % PALETTE.length]; break; }
      const hatch = dim && dim.some((m) => m[i]) && (((i % RW) + ((i / RW) | 0)) % 8 < 3);
      for (let k = 0; k < 3; k++) {
        rgb[i * 3 + k] = c ? Math.round(photo[i * 3 + k] * 0.4 + c[k] * 0.6)
          : hatch ? 255 : Math.round(photo[i * 3 + k] * 0.45);
      }
    }
    this.save(name, rgb, RW, RH);
  }

  writeJson(name, obj) {
    fs.writeFileSync(path.join(this.dir, name), JSON.stringify(obj, null, 1));
    this.files.push(name);
  }

  async flush() {
    await Promise.all(this.pending);
    this.pending = [];
    return this.files;
  }
}
