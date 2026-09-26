/**
 * Compact QR encoder: byte mode, error-correction level M, versions 1-10.
 *
 * That range covers ~270 bytes, comfortably more than any share URL this app
 * produces, and keeps the whole thing to one small dependency-free module.
 */

// --------------------------------------------------------------- GF(256) ---

const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
(() => {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
})();

const gmul = (a, b) => (a === 0 || b === 0 ? 0 : EXP[LOG[a] + LOG[b]]);

function rsGenerator(degree) {
  let poly = [1];
  for (let i = 0; i < degree; i++) {
    const next = new Array(poly.length + 1).fill(0);
    for (let j = 0; j < poly.length; j++) {
      next[j] ^= gmul(poly[j], 1);
      next[j + 1] ^= gmul(poly[j], EXP[i]);
    }
    poly = next;
  }
  return poly;
}

function rsEncode(data, ecLen) {
  const gen = rsGenerator(ecLen);
  const res = new Uint8Array(data.length + ecLen);
  res.set(data);
  for (let i = 0; i < data.length; i++) {
    const factor = res[i];
    if (!factor) continue;
    for (let j = 0; j < gen.length; j++) res[i + j] ^= gmul(gen[j], factor);
  }
  return res.slice(data.length);
}

// -------------------------------------------------------- version tables ---

// Per version (1-10) at ECC level M: [totalCodewords, ecPerBlock, group1Blocks,
// group1DataCodewords, group2Blocks, group2DataCodewords]
const VERSIONS = {
  1:  [26, 10, 1, 16, 0, 0],
  2:  [44, 16, 1, 28, 0, 0],
  3:  [70, 26, 1, 44, 0, 0],
  4:  [100, 18, 2, 32, 0, 0],
  5:  [134, 24, 2, 43, 0, 0],
  6:  [172, 16, 4, 27, 0, 0],
  7:  [196, 18, 4, 31, 0, 0],
  8:  [242, 22, 2, 38, 2, 39],
  9:  [292, 22, 3, 36, 2, 37],
  10: [346, 26, 4, 43, 1, 44],
};

const ALIGN_POS = {
  1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
  6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
};

// -------------------------------------------------------------- bit stream --

class Bits {
  constructor() { this.bits = []; }
  push(value, length) {
    for (let i = length - 1; i >= 0; i--) this.bits.push((value >> i) & 1);
  }
  get length() { return this.bits.length; }
  toBytes() {
    while (this.bits.length % 8) this.bits.push(0);
    const out = new Uint8Array(this.bits.length / 8);
    for (let i = 0; i < out.length; i++) {
      let b = 0;
      for (let j = 0; j < 8; j++) b = (b << 1) | this.bits[i * 8 + j];
      out[i] = b;
    }
    return out;
  }
}

// ------------------------------------------------------------------ encode --

export function encode(text) {
  const data = new TextEncoder().encode(String(text ?? ''));

  let version = 0;
  let spec = null;
  for (let v = 1; v <= 10; v++) {
    const [total, ec, g1, d1, g2, d2] = VERSIONS[v];
    const capacity = g1 * d1 + g2 * d2;
    const header = 4 + (v < 10 ? 8 : 16);
    if (header + data.length * 8 <= capacity * 8) { version = v; spec = VERSIONS[v]; break; }
  }
  if (!version) return null;

  const [, ecLen, g1, d1, g2, d2] = spec;
  const dataCodewords = g1 * d1 + g2 * d2;

  const bits = new Bits();
  bits.push(0b0100, 4);                              // byte mode
  bits.push(data.length, version < 10 ? 8 : 16);
  for (const b of data) bits.push(b, 8);
  // Terminator, then pad alternately with the two standard pad bytes.
  const capBits = dataCodewords * 8;
  bits.push(0, Math.min(4, capBits - bits.length));
  const bytes = Array.from(bits.toBytes());
  const PADS = [0xec, 0x11];
  for (let i = 0; bytes.length < dataCodewords; i++) bytes.push(PADS[i % 2]);

  // Split into blocks, compute EC per block, then interleave both.
  const blocks = [];
  let off = 0;
  for (let i = 0; i < g1; i++) { blocks.push(bytes.slice(off, off + d1)); off += d1; }
  for (let i = 0; i < g2; i++) { blocks.push(bytes.slice(off, off + d2)); off += d2; }
  const ecBlocks = blocks.map((b) => rsEncode(Uint8Array.from(b), ecLen));

  const final = [];
  const maxData = Math.max(...blocks.map((b) => b.length));
  for (let i = 0; i < maxData; i++) {
    for (const b of blocks) if (i < b.length) final.push(b[i]);
  }
  for (let i = 0; i < ecLen; i++) for (const b of ecBlocks) final.push(b[i]);

  return buildMatrix(version, final);
}

// ------------------------------------------------------------ matrix build --

function buildMatrix(version, codewords) {
  const size = version * 4 + 17;
  const m = Array.from({ length: size }, () => new Array(size).fill(null));
  const reserved = Array.from({ length: size }, () => new Array(size).fill(false));

  const setFn = (x, y, v) => {
    if (x < 0 || y < 0 || x >= size || y >= size) return;
    m[y][x] = v;
    reserved[y][x] = true;
  };

  const finder = (cx, cy) => {
    for (let dy = -1; dy <= 7; dy++) {
      for (let dx = -1; dx <= 7; dx++) {
        const inRing = (dx >= 0 && dx <= 6 && (dy === 0 || dy === 6))
          || (dy >= 0 && dy <= 6 && (dx === 0 || dx === 6));
        const inCore = dx >= 2 && dx <= 4 && dy >= 2 && dy <= 4;
        setFn(cx + dx, cy + dy, inRing || inCore ? 1 : 0);
      }
    }
  };
  finder(0, 0);
  finder(size - 7, 0);
  finder(0, size - 7);

  // Timing patterns.
  for (let i = 8; i < size - 8; i++) {
    setFn(i, 6, i % 2 === 0 ? 1 : 0);
    setFn(6, i, i % 2 === 0 ? 1 : 0);
  }

  // Alignment patterns, skipping the ones that collide with finders.
  const pos = ALIGN_POS[version];
  for (const ax of pos) {
    for (const ay of pos) {
      const nearFinder = (ax < 9 && ay < 9)
        || (ax > size - 10 && ay < 9)
        || (ax < 9 && ay > size - 10);
      if (nearFinder) continue;
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const ring = Math.max(Math.abs(dx), Math.abs(dy));
          setFn(ax + dx, ay + dy, ring === 1 ? 0 : 1);
        }
      }
    }
  }

  setFn(8, size - 8, 1); // dark module

  // Reserve the format-information strips before laying data.
  for (let i = 0; i < 9; i++) {
    if (m[8][i] === null) { m[8][i] = 0; reserved[8][i] = true; }
    if (m[i][8] === null) { m[i][8] = 0; reserved[i][8] = true; }
  }
  for (let i = 0; i < 8; i++) {
    if (m[8][size - 1 - i] === null) { m[8][size - 1 - i] = 0; reserved[8][size - 1 - i] = true; }
    if (m[size - 1 - i][8] === null) { m[size - 1 - i][8] = 0; reserved[size - 1 - i][8] = true; }
  }

  // Zig-zag data placement, right to left, skipping the vertical timing column.
  let bitIndex = 0;
  const nextBit = () => {
    const byte = codewords[bitIndex >> 3];
    const bit = byte === undefined ? 0 : (byte >> (7 - (bitIndex & 7))) & 1;
    bitIndex++;
    return bit;
  };

  let upward = true;
  for (let col = size - 1; col > 0; col -= 2) {
    if (col === 6) col--;
    for (let i = 0; i < size; i++) {
      const row = upward ? size - 1 - i : i;
      for (const c of [col, col - 1]) {
        if (reserved[row][c]) continue;
        m[row][c] = nextBit();
      }
    }
    upward = !upward;
  }

  // Pick the mask that scores best, then write the matching format bits.
  let best = null;
  for (let mask = 0; mask < 8; mask++) {
    const cand = applyMask(m, reserved, mask, size);
    writeFormat(cand, mask, size);
    const score = penalty(cand, size);
    if (!best || score < best.score) best = { score, matrix: cand };
  }
  return best.matrix.map((row) => row.map((v) => v === 1));
}

const MASKS = [
  (r, c) => (r + c) % 2 === 0,
  (r) => r % 2 === 0,
  (r, c) => c % 3 === 0,
  (r, c) => (r + c) % 3 === 0,
  (r, c) => (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0,
  (r, c) => ((r * c) % 2) + ((r * c) % 3) === 0,
  (r, c) => (((r * c) % 2) + ((r * c) % 3)) % 2 === 0,
  (r, c) => (((r + c) % 2) + ((r * c) % 3)) % 2 === 0,
];

function applyMask(m, reserved, mask, size) {
  const out = m.map((row) => row.slice());
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (reserved[r][c]) continue;
      if (MASKS[mask](r, c)) out[r][c] ^= 1;
    }
  }
  return out;
}

function writeFormat(m, mask, size) {
  // Level M is 0b00; append the mask, then the BCH(15,5) remainder.
  const data = (0b00 << 3) | mask;
  let rem = data;
  for (let i = 0; i < 10; i++) {
    rem <<= 1;
    if (rem & 0x400) rem ^= 0x537;
  }
  const bitsVal = (((data << 10) | rem) ^ 0x5412) & 0x7fff;
  const bit = (i) => (bitsVal >> i) & 1;

  for (let i = 0; i <= 5; i++) m[8][i] = bit(i);
  m[8][7] = bit(6);
  m[8][8] = bit(7);
  m[7][8] = bit(8);
  for (let i = 9; i <= 14; i++) m[14 - i][8] = bit(i);

  for (let i = 0; i <= 7; i++) m[size - 1 - i][8] = bit(i);
  for (let i = 8; i <= 14; i++) m[8][size - 15 + i] = bit(i);
  m[size - 8][8] = 1;
}

function penalty(m, size) {
  let score = 0;

  // Rule 1: runs of five or more identical modules.
  const run = (get) => {
    for (let a = 0; a < size; a++) {
      let count = 1;
      for (let b = 1; b < size; b++) {
        if (get(a, b) === get(a, b - 1)) count++;
        else { if (count >= 5) score += 3 + (count - 5); count = 1; }
      }
      if (count >= 5) score += 3 + (count - 5);
    }
  };
  run((r, c) => m[r][c]);
  run((c, r) => m[r][c]);

  // Rule 2: 2x2 blocks of one colour.
  for (let r = 0; r < size - 1; r++) {
    for (let c = 0; c < size - 1; c++) {
      const v = m[r][c];
      if (v === m[r][c + 1] && v === m[r + 1][c] && v === m[r + 1][c + 1]) score += 3;
    }
  }

  // Rule 4: deviation from a 50/50 dark ratio.
  let dark = 0;
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) dark += m[r][c];
  const pct = (dark * 100) / (size * size);
  score += Math.floor(Math.abs(pct - 50) / 5) * 10;

  return score;
}
