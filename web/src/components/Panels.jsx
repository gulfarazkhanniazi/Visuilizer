import { useMemo, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  setActiveSurface, updateSurface, applyProduct, setLinkSameType,
} from '../store/vizSlice.js';
import { toggleWishlist } from '../store/catalogSlice.js';
import { track } from '../api/client.js';
import { useMoney } from './Dialogs.jsx';
import {
  LAYOUTS, TILE_SIZES, GROUT_SIZES, GROUT_COLORS, SURFACE_BY_KEY,
  PAINT_COLORS, MODEL_CONTROLS, MATERIAL_BY_KEY, materialModel,
} from '../engine/layouts.js';
import { Slider, Switch, Section, Empty } from './ui.jsx';
import { IconLink, IconTiles, IconHeart, IconHeartFilled } from './Icons.jsx';
import { useT } from '../i18n/index.jsx';

/* ------------------------------------------------------------- surfaces --- */

export function SurfacePanel() {
  const dispatch = useDispatch();
  const room = useSelector((s) => s.viz.room);
  const active = useSelector((s) => s.viz.activeSurface);
  const frame = useSelector((s) => s.viz.activeFrame);
  const states = useSelector((s) => s.viz.frames[frame]);
  const products = useSelector((s) => s.catalog.products);
  const link = useSelector((s) => s.viz.linkSameType);
  const t = useT();

  return (
    <>
      <Section title={t('surfaces.title')}>
        <div className="surface-list">
          {room.objectList.map((o) => {
            const st = states[o.name];
            const product = products.find((p) => p.id === st?.productId);
            return (
              <button
                key={o.name}
                className={`surface-item ${active === o.name ? 'active' : ''}`}
                onClick={() => dispatch(setActiveSurface(o.name))}
              >
                <span
                  className="sw"
                  style={product ? { backgroundImage: `url(${product.thumb})` } : undefined}
                />
                <span className="txt">
                  <strong>{o.label ?? o.name}</strong>
                  <span>{product ? product.name : t('surfaces.notApplied')}</span>
                </span>
                <span
                  className="switch-wrap"
                  onClick={(e) => {
                    e.stopPropagation();
                    dispatch(updateSurface({
                      surface: o.name,
                      patch: { visible: st?.visible === false },
                    }));
                  }}
                >
                  <span className={`switch ${st?.visible === false ? '' : 'on'}`} />
                </span>
              </button>
            );
          })}
        </div>
      </Section>

      <Section title={t('surfaces.behaviour')}>
        <div className="row-between" style={{ marginBottom: 8 }}>
          <span className="tiny muted row" style={{ gap: 6 }}>
            <IconLink /> {t('surfaces.link')}
          </span>
          <button
            className={`switch ${link ? 'on' : ''}`}
            onClick={() => dispatch(setLinkSameType(!link))}
            aria-label="Link surfaces of the same type"
          />
        </div>
        <p className="tiny dim">{t('surfaces.linkHelp')}</p>
      </Section>
    </>
  );
}

/* -------------------------------------------------------------- material --- */

/**
 * Which rendering model the active surface is showing.
 *
 * The model is mirrored into the surface state when a product is applied, but
 * a scheme restored from an old share link predates that field, so the product
 * is the fallback source of truth.
 */
export function useSurfaceModel() {
  const state = useSelector((s) => s.viz.frames[s.viz.activeFrame]?.[s.viz.activeSurface]);
  const product = useSelector((s) => s.catalog.products.find((p) => p.id === state?.productId));
  if (!state?.productId) return 'module';
  return state.model ?? materialModel(product?.material);
}

export const controlsFor = (model) => MODEL_CONTROLS[model] ?? MODEL_CONTROLS.module;

/* ---------------------------------------------------------------- tiles --- */

export function TilePanel() {
  const dispatch = useDispatch();
  const products = useSelector((s) => s.catalog.products);
  const categories = useSelector((s) => s.catalog.productCategories);
  const object = useSelector((s) => s.viz.room?.objectList.find((o) => o.name === s.viz.activeSurface));
  const state = useSelector((s) => s.viz.frames[s.viz.activeFrame][s.viz.activeSurface]);
  const roomId = useSelector((s) => s.viz.room?.id);

  const vendor = useSelector((s) => s.catalog.vendor);
  const wishlist = useSelector((s) => s.catalog.wishlist);
  const money = useMoney();
  const t = useT();

  const [category, setCategory] = useState('all');
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('default');
  const [onlySaved, setOnlySaved] = useState(false);

  const surfaceKey = object?.product_surface;
  const showPrice = vendor?.settings?.allowPrice !== false;
  const showWishlist = vendor?.settings?.allowWishlist !== false;

  const visible = useMemo(() => {
    const list = products.filter((p) => {
      if (surfaceKey && !p.surfaces.includes(surfaceKey)) return false;
      if (category !== 'all' && p.category !== category) return false;
      if (onlySaved && !wishlist.includes(p.id)) return false;
      if (query && !`${p.name} ${p.sku ?? ''}`.toLowerCase().includes(query.toLowerCase())) return false;
      return true;
    });
    const byPrice = (dir) => (a, b) => {
      // Unpriced products sort last either way, rather than reading as free.
      if (a.price == null) return 1;
      if (b.price == null) return -1;
      return (a.price - b.price) * dir;
    };
    if (sort === 'name') list.sort((a, b) => a.name.localeCompare(b.name));
    if (sort === 'name-desc') list.sort((a, b) => b.name.localeCompare(a.name));
    if (sort === 'price-low') list.sort(byPrice(1));
    if (sort === 'price-high') list.sort(byPrice(-1));
    return list;
  }, [products, surfaceKey, category, query, sort, onlySaved, wishlist]);

  if (!object) return null;

  return (
    <>
      <input
        className="input"
        placeholder={t('products.search')}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ marginBottom: 8 }}
      />

      <div className="row" style={{ marginBottom: 10 }}>
        <select className="select grow" value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="default">{t('products.sortFeatured')}</option>
          <option value="name">{t('products.sortNameAsc')}</option>
          <option value="name-desc">{t('products.sortNameDesc')}</option>
          {showPrice && <option value="price-low">{t('products.sortPriceLow')}</option>}
          {showPrice && <option value="price-high">{t('products.sortPriceHigh')}</option>}
        </select>
        {showWishlist && (
          <button
            className={`chip ${onlySaved ? 'active' : ''}`}
            onClick={() => setOnlySaved((v) => !v)}
            title={t('products.savedOnly')}
          >
            <IconHeart /> {wishlist.length}
          </button>
        )}
      </div>

      <div className="filters" style={{ marginBottom: 12 }}>
        <button className={`chip ${category === 'all' ? 'active' : ''}`} onClick={() => setCategory('all')}>
          {t('products.all')}
        </button>
        {categories.map((c) => (
          <button
            key={c.id}
            className={`chip ${category === c.id ? 'active' : ''}`}
            onClick={() => setCategory(c.id)}
          >
            {c.name}
          </button>
        ))}
      </div>

      {visible.length ? (
        <div className="tile-grid">
          {visible.map((p) => (
            <div key={p.id} className={`tile-card ${state?.productId === p.id ? 'active' : ''}`}>
              <button
                className="tile-hit"
                onClick={() => {
                  dispatch(applyProduct({ product: p }));
                  track('product_apply', { roomId, productId: p.id, surface: object?.product_surface });
                }}
                title={`${p.name}${p.sku ? ` · ${p.sku}` : ''}`}
              >
                <span className="swatch" style={{ backgroundImage: `url(${p.thumb})` }} />
                <span className="name">{p.name}</span>
                {showPrice && p.price != null && (
                  <span className="price">{money(p.price)}<span className="dim">/{p.priceUnit}</span></span>
                )}
              </button>
              {showWishlist && (
                <button
                  className={`fav ${wishlist.includes(p.id) ? 'on' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    dispatch(toggleWishlist(p.id));
                    if (!wishlist.includes(p.id)) track('wishlist_add', { productId: p.id, roomId });
                  }}
                  title={wishlist.includes(p.id) ? t('products.unsave') : t('products.save')}
                  aria-label="Save product"
                >
                  {wishlist.includes(p.id) ? <IconHeartFilled /> : <IconHeart />}
                </button>
              )}
            </div>
          ))}
        </div>
      ) : (
        <Empty
          icon={<IconTiles />}
          title={t('products.emptyTitle')}
          hint={t('products.emptyHint', { surface: SURFACE_BY_KEY[surfaceKey]?.name ?? surfaceKey })}
        />
      )}
    </>
  );
}

/* --------------------------------------------------------------- layout --- */

/** Sensible preset sizes for the models that are not sold as tiles. */
const SHEET_REPEATS = [
  { w: 530, h: 1000, label: '530 × 1000' },
  { w: 700, h: 1000, label: '700 × 1000' },
  { w: 1000, h: 1000, label: '1000 × 1000' },
  { w: 1400, h: 1400, label: '1400 × 1400' },
  { w: 2000, h: 2000, label: '2000 × 2000' },
  { w: 3000, h: 3000, label: '3000 × 3000' },
];

const RUG_SIZES = [
  { w: 1200, h: 1700, label: '120 × 170' },
  { w: 1600, h: 2300, label: '160 × 230' },
  { w: 2000, h: 1400, label: '200 × 140' },
  { w: 2000, h: 3000, label: '200 × 300' },
  { w: 2400, h: 3400, label: '240 × 340' },
  { w: 3000, h: 4000, label: '300 × 400' },
];

function SizeGrid({ options, value, onPick }) {
  return (
    <div className="size-grid">
      {options.map((sz) => (
        <button
          key={`${sz.w}x${sz.h}`}
          className={`opt ${value.w === sz.w && value.h === sz.h ? 'active' : ''}`}
          onClick={() => onPick({ w: sz.w, h: sz.h })}
        >
          {sz.label ?? `${sz.w} × ${sz.h}`}
        </button>
      ))}
    </div>
  );
}

function SizeInputs({ value, onChange, max = 6000 }) {
  return (
    <div className="row" style={{ marginTop: 8 }}>
      <input
        className="input"
        type="number"
        min="20"
        max={max}
        value={value.w}
        onChange={(e) => onChange({ ...value, w: Number(e.target.value) || 1 })}
      />
      <span className="dim">×</span>
      <input
        className="input"
        type="number"
        min="20"
        max={max}
        value={value.h}
        onChange={(e) => onChange({ ...value, h: Number(e.target.value) || 1 })}
      />
    </div>
  );
}

/**
 * How the material is laid out on the plane.
 *
 * Which controls make sense depends entirely on the rendering model: a bond
 * pattern is meaningless for wallpaper, a repeat length is meaningless for a
 * rug, and a rug needs positioning where a tile never does.
 */
export function LayoutPanel() {
  const dispatch = useDispatch();
  const state = useSelector((s) => s.viz.frames[s.viz.activeFrame][s.viz.activeSurface]);
  const products = useSelector((s) => s.catalog.products);
  const product = products.find((p) => p.id === state?.productId);
  const model = useSurfaceModel();
  const controls = controlsFor(model);
  const t = useT();

  if (!state) return null;
  const set = (patch) => dispatch(updateSurface({ patch }));
  const live = (patch) => dispatch(updateSurface({ patch, skipHistory: true }));
  const offset = state.offset ?? { x: 0, y: 0 };

  const sizeTitle = t(
    model === 'sheet' ? 'layout.repeat'
      : model === 'piece' ? 'layout.rugSize'
        : model === 'joint' ? 'layout.existingSize'
          : 'layout.tileSize',
  );

  const presets = model === 'sheet' ? SHEET_REPEATS
    : model === 'piece' ? RUG_SIZES
      : (product?.sizes?.length && model !== 'joint' ? product.sizes : TILE_SIZES);

  return (
    <>
      {controls.size && (
        <Section title={sizeTitle}>
          <SizeGrid options={presets} value={state.tileSize} onPick={(v) => set({ tileSize: v })} />
          <SizeInputs value={state.tileSize} onChange={(v) => set({ tileSize: v })} />
          {model === 'sheet' && (
            <p className="tiny dim" style={{ marginTop: 6 }}>{t('layout.repeatHelp')}</p>
          )}
          {model === 'joint' && (
            <p className="tiny dim" style={{ marginTop: 6 }}>{t('layout.jointHelp')}</p>
          )}
        </Section>
      )}

      {controls.bond && (
        <Section title={t('layout.bond')}>
          <div className="layout-grid">
            {LAYOUTS.map((l) => (
              <button
                key={l.key}
                className={`opt ${state.layout === l.key ? 'active' : ''}`}
                onClick={() => set({ layout: l.key })}
                title={l.name}
              >
                <LayoutGlyph kind={l.key} />
                <span style={{ display: 'block', marginTop: 3 }}>{l.name}</span>
              </button>
            ))}
          </div>
        </Section>
      )}

      <Section title={t(controls.place ? 'layout.position' : 'layout.orientation')}>
        {!controls.place && (
          <div className="row" style={{ gap: 6, marginBottom: 12 }}>
            {[0, 90, 180, 270].map((deg) => (
              <button
                key={deg}
                className={`opt grow ${state.rotation === deg ? 'active' : ''}`}
                onClick={() => set({ rotation: deg })}
              >
                {deg}°
              </button>
            ))}
          </div>
        )}
        <Slider
          label={t(controls.place ? 'layout.rotation' : 'layout.fineRotation')}
          value={state.rotation}
          min={0}
          max={359}
          step={1}
          unit="°"
          onChange={(v) => live({ rotation: v })}
        />
        <Slider
          label={t(controls.place ? 'layout.moveX' : 'layout.shiftX')}
          value={offset.x}
          min={controls.place ? -4 : -2}
          max={controls.place ? 4 : 2}
          step={0.01}
          onChange={(v) => live({ offset: { ...offset, x: v } })}
          format={(v) => `${(v * 1000).toFixed(0)} mm`}
        />
        <Slider
          label={t(controls.place ? 'layout.moveY' : 'layout.shiftY')}
          value={offset.y}
          min={controls.place ? -4 : -2}
          max={controls.place ? 4 : 2}
          step={0.01}
          onChange={(v) => live({ offset: { ...offset, y: v } })}
          format={(v) => `${(v * 1000).toFixed(0)} mm`}
        />
        {controls.place && (
          <div className="row" style={{ marginTop: 4 }}>
            <button
              className="btn btn-sm"
              onClick={() => set({ offset: { x: 0, y: 0 }, rotation: 0 })}
            >
              {t('layout.centre')}
            </button>
          </div>
        )}
      </Section>

      {controls.variation && (
        <Section title={t('layout.variation')}>
          <Switch
            label={t('layout.randomFace')}
            checked={!!state.randomFace}
            onChange={(v) => set({ randomFace: v })}
          />
          <Switch
            label={t('layout.randomRotate')}
            checked={!!state.randomRotate}
            onChange={(v) => set({ randomRotate: v })}
          />
          <p className="tiny dim">
            {product?.faces?.length > 1
              ? t('layout.facesAvailable', { n: product.faces.length })
              : t('layout.singleFace')}
          </p>
        </Section>
      )}
    </>
  );
}

/* ---------------------------------------------------------------- paint --- */

/**
 * Paint is a colour, not a texture. The shader takes it directly rather than
 * tinting an image, so a change is instant and the swatch is exact.
 */
export function PaintPanel() {
  const dispatch = useDispatch();
  const state = useSelector((s) => s.viz.frames[s.viz.activeFrame][s.viz.activeSurface]);
  const t = useT();
  if (!state) return null;
  const set = (patch, skipHistory = false) => dispatch(updateSurface({ patch, skipHistory }));
  const color = (state.color ?? '#eae3d6').toLowerCase();

  return (
    <>
      <Section title={t('paint.colour')}>
        <div className="paint-grid">
          {PAINT_COLORS.map((c) => (
            <button
              key={c.hex}
              className={`paint-chip ${color === c.hex ? 'active' : ''}`}
              style={{ background: c.hex }}
              title={c.name}
              onClick={() => set({ color: c.hex })}
            >
              <span>{c.name}</span>
            </button>
          ))}
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <input
            type="color"
            value={color}
            onChange={(e) => set({ color: e.target.value }, true)}
            style={{ width: 44, height: 32, background: 'none', border: 'none', cursor: 'pointer' }}
          />
          <input
            className="input grow"
            value={color}
            onChange={(e) => set({ color: e.target.value }, true)}
          />
        </div>
        <p className="tiny dim" style={{ marginTop: 8 }}>
          {t('paint.help', {
            name: PAINT_COLORS.find((c) => c.hex === color)?.name ?? t('paint.custom'),
          })}
        </p>
      </Section>

      <Section title={t('paint.sheen')}>
        <div className="row" style={{ gap: 6 }}>
          {[['paint.matt', 0.03], ['paint.eggshell', 0.12], ['paint.satin', 0.26], ['paint.gloss', 0.5]].map(([label, g]) => (
            <button
              key={label}
              className={`opt grow ${Math.abs((state.gloss ?? 0) - g) < 0.04 ? 'active' : ''}`}
              onClick={() => set({ gloss: g })}
            >
              {t(label)}
            </button>
          ))}
        </div>
      </Section>
    </>
  );
}

/* ---------------------------------------------------------------- grout --- */

export function GroutPanel() {
  const dispatch = useDispatch();
  const state = useSelector((s) => s.viz.frames[s.viz.activeFrame][s.viz.activeSurface]);
  const model = useSurfaceModel();
  const t = useT();
  if (!state) return null;

  const set = (patch) => dispatch(updateSurface({ patch }));
  const grout = state.grout ?? { size: 2, color: '#c9c9c4' };

  return (
    <>
      <Section title={t('grout.width')}>
        <div className="size-grid">
          {GROUT_SIZES.map((g) => (
            <button
              key={g}
              className={`opt ${grout.size === g ? 'active' : ''}`}
              onClick={() => set({ grout: { ...grout, size: g } })}
            >
              {g === 0 ? t('grout.none') : `${g} mm`}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 10 }}>
          <Slider
            label={t('grout.custom')}
            value={grout.size}
            min={0}
            max={15}
            step={0.5}
            unit=" mm"
            onChange={(v) => dispatch(updateSurface({
              patch: { grout: { ...grout, size: v } }, skipHistory: true,
            }))}
          />
        </div>
      </Section>

      <Section title={t('grout.colour')}>
        <div className="swatch-row">
          {GROUT_COLORS.map((c) => (
            <button
              key={c.hex}
              className={`swatch-dot ${grout.color === c.hex ? 'active' : ''}`}
              style={{ background: c.hex }}
              title={c.name}
              onClick={() => set({ grout: { ...grout, color: c.hex } })}
            />
          ))}
          <label
            className="swatch-dot"
            style={{
              background: 'conic-gradient(red, yellow, lime, aqua, blue, magenta, red)',
              display: 'grid',
              placeItems: 'center',
              cursor: 'pointer',
            }}
            title={t('grout.customColour')}
          >
            <input
              type="color"
              value={grout.color}
              onChange={(e) => set({ grout: { ...grout, color: e.target.value } })}
              style={{ opacity: 0, width: 0, height: 0 }}
            />
          </label>
        </div>
        <p className="tiny dim" style={{ marginTop: 8 }}>
          {t('grout.currently', {
            name: GROUT_COLORS.find((c) => c.hex === grout.color)?.name ?? grout.color,
          })}
        </p>
      </Section>

      {model === 'joint' ? (
        <div className="hint">{t('grout.jointOnly')}</div>
      ) : (
        <Section title={t('grout.edge')}>
          <Slider
            label={t('grout.bevel')}
            value={state.bevel ?? 0.35}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => dispatch(updateSurface({ patch: { bevel: v }, skipHistory: true }))}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <p className="tiny dim">{t('grout.bevelHelp')}</p>
        </Section>
      )}
    </>
  );
}

/* --------------------------------------------------------------- finish --- */

export function FinishPanel() {
  const dispatch = useDispatch();
  const state = useSelector((s) => s.viz.frames[s.viz.activeFrame][s.viz.activeSurface]);
  const t = useT();
  if (!state) return null;
  const set = (patch, skipHistory = true) => dispatch(updateSurface({ patch, skipHistory }));

  return (
    <>
      <Section title={t('finish.title')}>
        <div className="row" style={{ gap: 6, marginBottom: 14 }}>
          {[['finish.matt', 0.06], ['finish.satin', 0.28], ['finish.gloss', 0.62], ['finish.polished', 0.9]].map(([label, g]) => (
            <button
              key={label}
              className={`opt grow ${Math.abs((state.gloss ?? 0) - g) < 0.05 ? 'active' : ''}`}
              onClick={() => set({ gloss: g }, false)}
            >
              {t(label)}
            </button>
          ))}
        </div>
        <Slider
          label={t('finish.reflectivity')}
          value={state.gloss ?? 0.25}
          min={0}
          max={1.2}
          step={0.01}
          onChange={(v) => set({ gloss: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
      </Section>

      <Section title={t('finish.blend')}>
        <Slider
          label={t('finish.keepLighting')}
          value={state.shade ?? 1}
          min={0}
          max={1.8}
          step={0.01}
          onChange={(v) => set({ shade: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <p className="tiny dim" style={{ marginTop: -6, marginBottom: 12 }}>
          {t('finish.keepLightingHelp')}
        </p>
        <Slider
          label={t('finish.detail')}
          value={state.detail ?? 0.6}
          min={0}
          max={1.5}
          step={0.01}
          onChange={(v) => set({ detail: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
        <p className="tiny dim" style={{ marginTop: -6 }}>{t('finish.detailHelp')}</p>
      </Section>

      <Section title={t('finish.tint')}>
        <div className="row">
          <input
            type="color"
            value={state.tint ?? '#ffffff'}
            onChange={(e) => set({ tint: e.target.value }, false)}
            style={{ width: 44, height: 32, background: 'none', border: 'none', cursor: 'pointer' }}
          />
          <button className="btn btn-sm" onClick={() => set({ tint: '#ffffff' }, false)}>
            {t('finish.resetTint')}
          </button>
          <span className="tiny dim grow">{t('finish.tintHelp')}</span>
        </div>
      </Section>
    </>
  );
}

/* ---------------------------------------------------------------- glyph --- */

/** Tiny previews of each bond pattern, drawn rather than shipped as images. */
function LayoutGlyph({ kind }) {
  const r = (x, y, w, h, k) => <rect key={k} x={x} y={y} width={w} height={h} rx="0.6" />;
  let shapes = [];
  if (kind === 'grid') {
    shapes = [r(1, 1, 8, 5, 'a'), r(10, 1, 8, 5, 'b'), r(1, 7, 8, 5, 'c'), r(10, 7, 8, 5, 'd')];
  } else if (kind === 'brick' || kind === 'diagonal-brick') {
    shapes = [r(1, 1, 8, 5, 'a'), r(10, 1, 8, 5, 'b'), r(-3, 7, 8, 5, 'c'), r(5.5, 7, 8, 5, 'd'), r(14, 7, 8, 5, 'e')];
  } else if (kind === 'brick-third') {
    shapes = [r(1, 1, 8, 5, 'a'), r(10, 1, 8, 5, 'b'), r(-2, 7, 8, 5, 'c'), r(7, 7, 8, 5, 'd'), r(16, 7, 8, 5, 'e')];
  } else if (kind === 'vertical') {
    shapes = [r(1, 1, 5, 11, 'a'), r(7, 1, 5, 11, 'b'), r(13, 1, 5, 11, 'c')];
  } else if (kind === 'vertical-brick') {
    shapes = [r(1, -2, 5, 8, 'a'), r(7, 1, 5, 8, 'b'), r(13, -2, 5, 8, 'c'), r(1, 7, 5, 8, 'd'), r(13, 7, 5, 8, 'e')];
  } else if (kind === 'herringbone') {
    return (
      <svg viewBox="0 0 20 13" width="26" height="17" fill="currentColor" opacity=".85">
        <g transform="rotate(45 10 6.5)">
          <rect x="1" y="2" width="8" height="3.4" rx=".5" />
          <rect x="9.5" y="-2.4" width="3.4" height="8" rx=".5" />
          <rect x="9.5" y="6" width="8" height="3.4" rx=".5" />
          <rect x="5.6" y="6" width="3.4" height="8" rx=".5" />
        </g>
      </svg>
    );
  } else if (kind === 'basketweave') {
    shapes = [
      r(1, 1, 7.5, 2.4, 'a'), r(1, 4, 7.5, 2.4, 'b'),
      r(10, 1, 2.4, 5.4, 'c'), r(13, 1, 2.4, 5.4, 'd'),
      r(1, 7.5, 2.4, 4.5, 'e'), r(4, 7.5, 2.4, 4.5, 'f'),
      r(10, 7.5, 7.5, 1.9, 'g'), r(10, 10, 7.5, 1.9, 'h'),
    ];
  } else if (kind === 'diagonal') {
    return (
      <svg viewBox="0 0 20 13" width="26" height="17" fill="currentColor" opacity=".85">
        <g transform="rotate(45 10 6.5)">
          <rect x="3" y="0" width="6" height="6" rx=".6" />
          <rect x="10" y="0" width="6" height="6" rx=".6" />
          <rect x="3" y="7" width="6" height="6" rx=".6" />
          <rect x="10" y="7" width="6" height="6" rx=".6" />
        </g>
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 20 13" width="26" height="17" fill="currentColor" opacity=".85">
      <clipPath id={`c-${kind}`}><rect x="0" y="0" width="20" height="13" /></clipPath>
      <g clipPath={`url(#c-${kind})`}>{shapes}</g>
    </svg>
  );
}
