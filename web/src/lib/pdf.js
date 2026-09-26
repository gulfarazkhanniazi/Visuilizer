/**
 * Branded PDF export.
 *
 * Two documents, both built in the browser from what is already on screen:
 *  - a room sheet: the rendered visualisation plus the products used on it;
 *  - a product catalogue: a grid of swatches with sizes, finishes and prices.
 *
 * jsPDF is loaded on demand, so the 150 kB never lands on visitors who only
 * ever look at a room.
 */

const A4 = { w: 210, h: 297 };   // mm, portrait
const M = 15;                    // page margin

async function newDoc(orientation = 'portrait') {
  const { jsPDF } = await import('jspdf');
  return new jsPDF({ unit: 'mm', format: 'a4', orientation, compress: true });
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '#2563eb');
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [37, 99, 235];
}

/** Shared header band, so both documents read as the same brand. */
async function header(doc, vendor, title, pageW) {
  const [r, g, b] = hexToRgb(vendor?.primaryColor);
  doc.setFillColor(r, g, b);
  doc.rect(0, 0, pageW, 26, 'F');

  if (vendor?.logo) {
    try {
      const data = await toDataUrl(vendor.logo);
      const { width, height } = await imageSize(data);
      const h = 12;
      doc.addImage(data, 'JPEG', M, 7, (width / height) * h, h);
    } catch { /* a missing logo must not stop the export */ }
  }

  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(15);
  doc.text(vendor?.name ?? 'Surface Studio', pageW - M, 13, { align: 'right' });
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.text(title, pageW - M, 19.5, { align: 'right' });
  doc.setTextColor(20, 20, 20);
}

function footer(doc, vendor, pageW, pageH) {
  doc.setDrawColor(220);
  doc.line(M, pageH - 14, pageW - M, pageH - 14);
  doc.setFontSize(8);
  doc.setTextColor(130);
  const contact = [vendor?.settings?.contactEmail, vendor?.settings?.contactPhone]
    .filter(Boolean).join('  ·  ');
  doc.text(contact || (vendor?.name ?? ''), M, pageH - 9);
  doc.text(new Date().toLocaleDateString(), pageW - M, pageH - 9, { align: 'right' });
  doc.setTextColor(20, 20, 20);
}

/**
 * Room sheet: the visualisation, then the products applied to it.
 * `imageDataUrl` comes straight from the renderer's full-resolution export.
 */
export async function roomSheetPdf({ vendor, room, imageDataUrl, products, money }) {
  const doc = await newDoc('portrait');
  await header(doc, vendor, 'Room visualisation', A4.w);

  let y = 34;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.text(room.name, M, y);
  y += 7;

  const imgW = A4.w - M * 2;
  const { width, height } = await imageSize(imageDataUrl);
  const imgH = (height / width) * imgW;
  doc.addImage(imageDataUrl, 'JPEG', M, y, imgW, imgH);
  y += imgH + 10;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.text('Products used', M, y);
  y += 6;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);

  for (const p of products) {
    if (y > A4.h - 30) { footer(doc, vendor, A4.w, A4.h); doc.addPage(); await header(doc, vendor, 'Room visualisation', A4.w); y = 34; }
    try {
      const swatch = await toDataUrl(p.thumb);
      doc.addImage(swatch, 'JPEG', M, y - 4, 16, 16);
    } catch { /* keep going without the swatch */ }

    doc.setFont('helvetica', 'bold');
    doc.text(p.name, M + 20, y + 1);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(110);
    const bits = [
      p.surface,
      p.sku,
      p.tileSize ? `${p.tileSize.w}x${p.tileSize.h} mm` : null,
      p.finish,
    ].filter(Boolean).join('   ·   ');
    doc.text(bits, M + 20, y + 6);
    if (p.price != null && money) {
      doc.setTextColor(20);
      doc.text(`${money(p.price)} / ${p.priceUnit}`, A4.w - M, y + 1, { align: 'right' });
    }
    doc.setTextColor(20);
    y += 20;
  }

  footer(doc, vendor, A4.w, A4.h);
  return doc;
}

/** Product catalogue: a swatch grid, three across. */
export async function catalogPdf({ vendor, products, money, title = 'Product catalogue' }) {
  const doc = await newDoc('portrait');
  await header(doc, vendor, title, A4.w);

  const cols = 3;
  const gap = 5;
  const cardW = (A4.w - M * 2 - gap * (cols - 1)) / cols;
  // Just tight enough that three rows clear the footer, so a 16-product
  // catalogue is two pages rather than three.
  const cardH = cardW + 20;

  let x = M;
  let y = 34;
  let col = 0;

  for (const p of products) {
    if (y + cardH > A4.h - 16) {
      footer(doc, vendor, A4.w, A4.h);
      doc.addPage();
      await header(doc, vendor, title, A4.w);
      x = M; y = 34; col = 0;
    }

    try {
      const swatch = await toDataUrl(p.thumb ?? p.image);
      doc.addImage(swatch, 'JPEG', x, y, cardW, cardW);
    } catch {
      doc.setFillColor(238);
      doc.rect(x, y, cardW, cardW, 'F');
    }
    doc.setDrawColor(225);
    doc.rect(x, y, cardW, cardW);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.text(clip(doc, p.name, cardW), x, y + cardW + 5);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(120);
    const sizes = (p.sizes ?? []).slice(0, 2).map((s) => `${s.w}x${s.h}`).join(', ');
    doc.text(clip(doc, [p.sku, sizes].filter(Boolean).join('  ·  '), cardW), x, y + cardW + 9.5);
    doc.text(clip(doc, `${p.material ?? ''} ${p.finish ?? ''}`.trim(), cardW), x, y + cardW + 13.5);
    if (p.price != null && money) {
      doc.setTextColor(20);
      doc.setFont('helvetica', 'bold');
      doc.text(`${money(p.price)}/${p.priceUnit}`, x, y + cardW + 18);
    }
    doc.setTextColor(20);

    col += 1;
    if (col === cols) { col = 0; x = M; y += cardH + gap; }
    else x += cardW + gap;
  }

  footer(doc, vendor, A4.w, A4.h);
  return doc;
}

/** Single-product spec sheet -- the reference calls this "download detail". */
export async function productSheetPdf({ vendor, product, money }) {
  const doc = await newDoc('portrait');
  await header(doc, vendor, 'Product details', A4.w);

  let y = 36;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.text(product.name, M, y);
  y += 8;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.setTextColor(110);
  if (product.sku) { doc.text(`SKU ${product.sku}`, M, y); y += 8; }
  doc.setTextColor(20);

  const imgW = A4.w - M * 2;
  try {
    const face = await toDataUrl(product.faces?.[0] ?? product.image);
    doc.addImage(face, 'JPEG', M, y, imgW, imgW * 0.55);
    y += imgW * 0.55 + 10;
  } catch { y += 4; }

  const rows = [
    ['Material', product.material],
    ['Finish', product.finish],
    ['Sizes (mm)', (product.sizes ?? []).map((s) => `${s.w} x ${s.h}`).join(', ')],
    ['Suitable for', (product.surfaces ?? []).join(', ')],
    ['Pieces per box', product.piecesPerBox],
    ['Coverage per box', product.coverageSqm ? `${product.coverageSqm} m2` : null],
    ['Price', product.price != null && money ? `${money(product.price)} / ${product.priceUnit}` : null],
  ].filter(([, v]) => v != null && v !== '');

  doc.setFontSize(10);
  for (const [k, v] of rows) {
    doc.setTextColor(120);
    doc.text(k, M, y);
    doc.setTextColor(20);
    doc.text(String(v), M + 50, y);
    y += 7;
  }

  if (product.description) {
    y += 4;
    doc.setTextColor(90);
    doc.setFontSize(9);
    doc.text(doc.splitTextToSize(product.description, A4.w - M * 2), M, y);
  }

  footer(doc, vendor, A4.w, A4.h);
  return doc;
}

/* ------------------------------------------------------------- helpers -- */

function clip(doc, text, maxW) {
  if (!text) return '';
  let out = String(text);
  while (doc.getTextWidth(out) > maxW && out.length > 3) out = out.slice(0, -2);
  return out.length < String(text).length ? `${out}…` : out;
}

/** Fetch an image and re-encode it as a JPEG data URL that jsPDF accepts. */
async function toDataUrl(url) {
  if (!url) throw new Error('no image');
  if (url.startsWith('data:')) return url;
  const img = await new Promise((resolve, reject) => {
    const i = new Image();
    i.crossOrigin = 'anonymous';
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = url;
  });
  const canvas = document.createElement('canvas');
  const max = 700;
  const scale = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
  canvas.width = Math.round(img.naturalWidth * scale);
  canvas.height = Math.round(img.naturalHeight * scale);
  canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.85);
}

function imageSize(dataUrl) {
  return new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve({ width: i.naturalWidth, height: i.naturalHeight });
    i.onerror = reject;
    i.src = dataUrl;
  });
}
