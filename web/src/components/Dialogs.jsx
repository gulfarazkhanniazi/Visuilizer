import { useEffect, useMemo, useState } from 'react';
import { useSelector } from 'react-redux';
import { api, track } from '../api/client.js';
import { Modal, Section, useToast } from './ui.jsx';
import { surfaceAreaSqm } from '../engine/masks.js';
import { surfaceHomography, invert3 } from '../engine/homography.js';
import { materialModel } from '../engine/layouts.js';
import { useT } from '../i18n/index.jsx';

/** Format a money amount with the vendor's currency symbol. */
export function useMoney() {
  const vendor = useSelector((s) => s.catalog.vendor);
  const symbol = vendor?.settings?.currencySymbol ?? '$';
  return (n) => (n == null ? '—' : `${symbol}${Number(n).toFixed(2)}`);
}

/* -------------------------------------------------------------- inquiry -- */

/**
 * Product enquiry. Sends the visitor's details together with what they were
 * actually looking at, so the reply can reference the exact scheme.
 */
export function InquiryDialog({ onClose, context }) {
  const toast = useToast();
  const t = useT();
  const vendor = useSelector((s) => s.catalog.vendor);
  const [form, setForm] = useState({ name: '', email: '', phone: '', message: '' });
  const [busy, setBusy] = useState(false);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function send() {
    if (!form.email && !form.phone) {
      return toast(t('enquiry.need'), 'error');
    }
    setBusy(true);
    try {
      await api.createLead({ ...form, context });
      track('lead', { roomId: context?.roomId });
      toast(t('enquiry.sent'), 'ok');
      onClose();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title={t('enquiry.title')}
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>{t('common.cancel')}</button>
          <button className="btn btn-primary" onClick={send} disabled={busy}>
            {busy ? t('enquiry.sending') : t('enquiry.send')}
          </button>
        </>
      }
    >
      {context?.products?.length > 0 && (
        <div className="hint">
          {t('enquiry.with', { products: context.products.map((p) => p.name).join(', ') })}
        </div>
      )}
      <div className="grid-2">
        <div className="field">
          <label>{t('enquiry.name')}</label>
          <input className="input" value={form.name} onChange={set('name')} />
        </div>
        <div className="field">
          <label>{t('enquiry.phone')}</label>
          <input className="input" value={form.phone} onChange={set('phone')} />
        </div>
      </div>
      <div className="field">
        <label>{t('enquiry.email')}</label>
        <input className="input" type="email" value={form.email} onChange={set('email')} />
      </div>
      <div className="field">
        <label>{t('enquiry.message')}</label>
        <textarea
          className="textarea"
          value={form.message}
          onChange={set('message')}
          placeholder={t('enquiry.placeholder')}
        />
      </div>
      {(vendor?.settings?.contactEmail || vendor?.settings?.contactPhone) && (
        <p className="tiny dim">
          {t('enquiry.direct')} {vendor.settings.contactEmail} {vendor.settings.contactPhone}
        </p>
      )}
    </Modal>
  );
}

/* ----------------------------------------------------------- calculator -- */

const SQFT_PER_SQM = 10.7639;

/**
 * Quantity calculator.
 *
 * The reference product asks you to type the floor area in. We already know the
 * mask and the homography for every surface, so each one can measure itself --
 * "Measure" fills the row from the actual marked area, occluders excluded.
 */
export function CalculatorDialog({ onClose }) {
  const room = useSelector((s) => s.viz.room);
  const frames = useSelector((s) => s.viz.frames);
  const activeFrame = useSelector((s) => s.viz.activeFrame);
  const products = useSelector((s) => s.catalog.products);
  const vendor = useSelector((s) => s.catalog.vendor);
  const money = useMoney();
  const t = useT();

  const [unit, setUnit] = useState('sqm');
  const [wastage, setWastage] = useState(vendor?.settings?.wastagePercent ?? 10);
  const [rows, setRows] = useState([]);

  // Seed one row per surface that actually has a product on it.
  useEffect(() => {
    if (!room) return;
    const seeded = room.objectList
      .filter((o) => frames[activeFrame]?.[o.name]?.productId)
      .map((o) => ({
        key: o.name,
        label: o.label ?? o.name,
        area: '',
        productId: frames[activeFrame][o.name].productId,
        tileSize: frames[activeFrame][o.name].tileSize,
      }));
    setRows(seeded.length ? seeded : [{ key: 'manual', label: 'Area 1', area: '', productId: products[0]?.id, tileSize: { w: 600, h: 600 } }]);
  }, [room, activeFrame, frames, products]);

  // Rooms may carry a lens correction; the homography and the area integral
  // both have to use the same one the renderer is drawing with.
  const lens = { k1: Number(room?.settings?.lensK1) || 0, width: room?.width, height: room?.height };

  function areaOf(obj) {
    const H = surfaceHomography(obj.quad, obj.realSize.w, obj.realSize.h, lens);
    const hInv = H ? invert3(H) : null;
    return surfaceAreaSqm(obj.mask, hInv, room.width, room.height, 3, lens.k1);
  }

  function measure(i) {
    track('measure', { roomId: room?.id });
    const row = rows[i];
    const obj = room.objectList.find((o) => o.name === row.key);
    if (!obj) return;
    const sqm = areaOf(obj);
    const value = unit === 'sqft' ? sqm * SQFT_PER_SQM : sqm;
    setRows((r) => r.map((x, k) => (k === i ? { ...x, area: value.toFixed(2) } : x)));
  }

  function measureAll() {
    setRows((r) => r.map((row) => {
      const obj = room.objectList.find((o) => o.name === row.key);
      if (!obj) return row;
      const sqm = areaOf(obj);
      return { ...row, area: ((unit === 'sqft' ? sqm * SQFT_PER_SQM : sqm)).toFixed(2) };
    }));
  }

  const results = useMemo(() => rows.map((row) => {
    const product = products.find((p) => p.id === row.productId);
    const raw = parseFloat(row.area);
    if (!product || !Number.isFinite(raw) || raw <= 0) return { row, product, empty: true };

    const sqm = unit === 'sqft' ? raw / SQFT_PER_SQM : raw;
    const withWaste = sqm * (1 + (Number(wastage) || 0) / 100);
    const model = materialModel(product.material);
    const pieceSqm = (row.tileSize.w / 1000) * (row.tileSize.h / 1000);

    // Only a modular material is bought by the piece. Wallpaper and epoxy are
    // bought by the roll or the tin against their stated coverage, a rug is
    // one object however big the floor is, and grout is not a field material
    // at all -- counting any of them in "tiles" would be a made-up number.
    let pieces = null;
    if (model === 'module') pieces = Math.ceil(withWaste / pieceSqm);
    else if (model === 'piece') pieces = 1;

    const perBox = product.piecesPerBox || null;
    let boxes = null;
    if (model === 'module' && perBox) boxes = Math.ceil(pieces / perBox);
    else if ((model === 'sheet' || model === 'solid') && product.coverageSqm) {
      boxes = Math.ceil(withWaste / product.coverageSqm);
    } else if (model === 'piece') boxes = 1;

    // Charge for what actually gets bought: whole boxes where they exist.
    const billedSqm = boxes && product.coverageSqm ? boxes * product.coverageSqm : withWaste;
    let cost = null;
    if (product.price != null) {
      if (model === 'piece') cost = product.priceUnit === 'pc' ? product.price : billedSqm * product.price;
      else if (model === 'joint') cost = null;
      else cost = billedSqm * product.price;
    }

    return { row, product, model, sqm, withWaste, pieces, perBox, boxes, billedSqm, cost };
  }), [rows, products, unit, wastage]);

  const total = results.reduce((n, r) => n + (r.cost ?? 0), 0);
  const anyPriced = results.some((r) => r.cost != null);
  const showPrice = vendor?.settings?.allowPrice !== false;

  return (
    <Modal title={t('calc.title')} onClose={onClose} width={720}
      footer={<button className="btn btn-primary" onClick={onClose}>{t('common.done')}</button>}>

      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field grow" style={{ marginBottom: 0 }}>
          <label>{t('calc.measureIn')}</label>
          <div className="row">
            {['sqm', 'sqft'].map((u) => (
              <button key={u} className={`opt grow ${unit === u ? 'active' : ''}`} onClick={() => setUnit(u)}>
                {u === 'sqm' ? t('calc.sqm') : t('calc.sqft')}
              </button>
            ))}
          </div>
        </div>
        <div className="field" style={{ marginBottom: 0, width: 130 }}>
          <label>{t('calc.wastage')}</label>
          <div className="row">
            <input className="input" type="number" min="0" max="40" value={wastage}
                   onChange={(e) => setWastage(e.target.value)} />
            <span className="dim">%</span>
          </div>
        </div>
        <button className="btn" style={{ marginTop: 18 }} onClick={measureAll}>{t('calc.measureAll')}</button>
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>{t('calc.area')}</th>
            <th style={{ width: 150 }}>{unit === 'sqm' ? t('calc.sqm') : t('calc.sqft')}</th>
            <th>{t('calc.product')}</th>
            <th style={{ width: 78 }}>{t('calc.units')}</th>
            <th style={{ width: 92 }}>{t('calc.boxes')}</th>
            {showPrice && <th style={{ width: 92 }}>{t('calc.cost')}</th>}
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={r.row.key + i}>
              <td>
                <strong>{r.row.label}</strong>
                <div className="tiny dim">{r.row.tileSize.w}×{r.row.tileSize.h} mm</div>
              </td>
              <td>
                <div className="row" style={{ gap: 4 }}>
                  <input
                    className="input"
                    type="number"
                    min="0"
                    step="0.01"
                    value={r.row.area}
                    placeholder="0.00"
                    onChange={(e) => setRows((rs) => rs.map((x, k) => (k === i ? { ...x, area: e.target.value } : x)))}
                  />
                  {r.row.key !== 'manual' && (
                    <button className="btn btn-sm" onClick={() => measure(i)} title={t('calc.measureHelp')}>
                      {t('calc.measure')}
                    </button>
                  )}
                </div>
              </td>
              <td className="muted">
                {r.product?.name ?? '—'}
                {r.product && <div className="tiny dim">{r.product.material}</div>}
              </td>
              <td>{r.empty ? '—' : (r.pieces ?? '—')}</td>
              <td>{r.empty ? '—' : (r.boxes ?? '—')}</td>
              {showPrice && <td>{r.empty ? '—' : money(r.cost)}</td>}
            </tr>
          ))}
        </tbody>
      </table>

      {showPrice && anyPriced && (
        <div className="row-between" style={{ marginTop: 14, fontSize: 16 }}>
          <strong>{t('calc.total')}</strong>
          <strong>{money(total)}</strong>
        </div>
      )}
      <p className="tiny dim" style={{ marginTop: 10 }}>{t('calc.note', { n: wastage })}</p>
    </Modal>
  );
}

/* ---------------------------------------------------------- save scheme -- */

export function SaveRoomDialog({ onClose, onSave, defaultName }) {
  const t = useT();
  const [name, setName] = useState(defaultName ?? 'My scheme');
  const [busy, setBusy] = useState(false);

  return (
    <Modal
      title={t('save.title')}
      onClose={onClose}
      footer={
        <>
          <button className="btn" onClick={onClose}>{t('common.cancel')}</button>
          <button
            className="btn btn-primary"
            disabled={busy}
            onClick={async () => { setBusy(true); await onSave(name); setBusy(false); }}
          >
            {busy ? t('save.saving') : t('save.save')}
          </button>
        </>
      }
    >
      <div className="field">
        <label>{t('save.name')}</label>
        <input className="input" value={name} autoFocus onChange={(e) => setName(e.target.value)} />
      </div>
      <p className="tiny dim">{t('save.help')}</p>
    </Modal>
  );
}
