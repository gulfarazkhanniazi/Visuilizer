import { useEffect, useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client.js';
import { loadCatalog, upsertProduct, removeProduct, setVendor } from '../store/catalogSlice.js';
import {
  SURFACE_TYPES, TILE_SIZES, MATERIALS, MATERIAL_BY_KEY, PAINT_COLORS, materialModel,
} from '../engine/layouts.js';
import { Modal, Section, Slider, Switch, useToast, Empty, QrCode } from '../components/ui.jsx';
import {
  IconPlus, IconTrash, IconEdit, IconUpload, IconRoom, IconTiles, IconStore, IconFile, IconCube,
} from '../components/Icons.jsx';
import AdminGate, { UsersTab } from './AdminAuth.jsx';
import AnalyticsTab from './AdminAnalytics.jsx';

const TABS = [
  'Analytics', 'Products', 'Rooms', 'Categories', 'Stores', 'Branding', 'Leads', 'Users',
];

export default function Admin() {
  return <AdminGate><AdminPanel /></AdminGate>;
}

function AdminPanel() {
  const [tab, setTab] = useState('Analytics');
  return (
    <main className="page">
      <div className="page-inner">
        <div className="page-head">
          <h1>Admin</h1>
          <p className="muted">Manage the catalogue, rooms, branding and enquiries.</p>
        </div>
        <div className="admin-tabs">
          {TABS.map((t) => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>
        {tab === 'Analytics' && <AnalyticsTab />}
        {tab === 'Products' && <ProductsTab />}
        {tab === 'Rooms' && <RoomsTab />}
        {tab === 'Categories' && <CategoriesTab />}
        {tab === 'Stores' && <StoresTab />}
        {tab === 'Branding' && <BrandingTab />}
        {tab === 'Leads' && <LeadsTab />}
        {tab === 'Users' && <UsersTab />}
      </div>
    </main>
  );
}

/* -------------------------------------------------------------- products -- */

function ProductsTab() {
  const dispatch = useDispatch();
  const toast = useToast();
  const products = useSelector((s) => s.catalog.products);
  const categories = useSelector((s) => s.catalog.productCategories);
  const vendor = useSelector((s) => s.catalog.vendor);
  const [editing, setEditing] = useState(null);
  const [query, setQuery] = useState('');
  const bulkRef = useRef(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  const shown = products.filter((p) =>
    !query || `${p.name} ${p.sku ?? ''}`.toLowerCase().includes(query.toLowerCase()));

  async function onBulk(e) {
    const files = [...(e.target.files ?? [])];
    e.target.value = '';
    if (!files.length) return;
    setBulkBusy(true);
    try {
      const res = await api.bulkProducts(files, { category: categories[0]?.id ?? '' });
      toast(`${res.created} products created.`, 'ok');
      dispatch(loadCatalog());
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      setBulkBusy(false);
    }
  }

  async function exportCatalog() {
    try {
      toast('Building the catalogue...');
      const { catalogPdf } = await import('../lib/pdf.js');
      const symbol = vendor?.settings?.currencySymbol ?? '$';
      const doc = await catalogPdf({
        vendor,
        products: shown,
        money: (n) => symbol + Number(n).toFixed(2),
      });
      doc.save('product-catalogue.pdf');
      toast('Catalogue downloaded.', 'ok');
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function onDelete(p) {
    if (!confirm(`Delete "${p.name}"? This cannot be undone.`)) return;
    try {
      await api.deleteProduct(p.id);
      dispatch(removeProduct(p.id));
      toast('Product deleted.', 'ok');
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  return (
    <>
      <div className="row" style={{ marginBottom: 16 }}>
        <input
          className="input grow"
          placeholder="Search products…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn" onClick={() => bulkRef.current?.click()} disabled={bulkBusy}>
          <IconUpload /> {bulkBusy ? 'Uploading…' : 'Bulk upload'}
        </button>
        <input ref={bulkRef} type="file" accept="image/*" multiple hidden onChange={onBulk} />
        <button className="btn" onClick={exportCatalog} disabled={!shown.length}>
          <IconFile /> Catalogue PDF
        </button>
        <button className="btn btn-primary" onClick={() => setEditing({})}>
          <IconPlus /> New product
        </button>
      </div>

      {shown.length ? (
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 56 }} />
              <th>Name</th>
              <th>SKU</th>
              <th>Category</th>
              <th>Sizes</th>
              <th>Surfaces</th>
              <th>Price</th>
              <th>Faces</th>
              <th style={{ width: 92 }} />
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => (
              <tr key={p.id}>
                <td><img src={p.thumb} alt="" /></td>
                <td><strong>{p.name}</strong></td>
                <td className="muted">{p.sku ?? '—'}</td>
                <td className="muted">{categories.find((c) => c.id === p.category)?.name ?? '—'}</td>
                <td className="muted tiny">{p.sizes.map((s) => `${s.w}×${s.h}`).join(', ')}</td>
                <td className="muted tiny">{p.surfaces.join(', ')}</td>
                <td className="muted">
                  {p.price != null
                    ? (vendor?.settings?.currencySymbol ?? '$') + p.price + '/' + p.priceUnit
                    : '—'}
                </td>
                <td className="muted">{p.faces.length}</td>
                <td>
                  <div className="row" style={{ gap: 4 }}>
                    <button className="btn btn-ghost btn-icon btn-sm" onClick={() => setEditing(p)}>
                      <IconEdit />
                    </button>
                    <button className="btn btn-ghost btn-icon btn-sm" onClick={() => onDelete(p)}>
                      <IconTrash />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty icon={<IconTiles />} title="No products yet" hint="Add one, or bulk-upload a folder of tile images." />
      )}

      {editing && (
        <ProductEditor
          product={editing}
          onClose={() => setEditing(null)}
          onSaved={(p) => { dispatch(upsertProduct(p)); setEditing(null); }}
        />
      )}
    </>
  );
}

function ProductEditor({ product, onClose, onSaved }) {
  const toast = useToast();
  const categories = useSelector((s) => s.catalog.productCategories);
  const isNew = !product.id;

  const [form, setForm] = useState({
    name: product.name ?? '',
    sku: product.sku ?? '',
    category: product.category ?? categories[0]?.id ?? '',
    material: product.material ?? 'tile',
    color: product.color ?? '#eae3d6',
    finish: product.finish ?? 'matt',
    gloss: product.gloss ?? 0.25,
    surfaces: product.surfaces ?? ['floor', 'wall'],
    sizes: product.sizes ?? [{ w: 600, h: 600 }],
    price: product.price ?? '',
    priceUnit: product.priceUnit ?? 'sqm',
    piecesPerBox: product.piecesPerBox ?? '',
    coverageSqm: product.coverageSqm ?? '',
    description: product.description ?? '',
  });
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);

  const model = materialModel(form.material);
  // Paint is a colour rather than a photograph, so it is the one material that
  // can be created without uploading anything.
  const needsFace = model !== 'solid';

  const set = (patch) => setForm((f) => ({ ...f, ...patch }));

  /** Switching material moves the default size onto something sane for it. */
  const setMaterial = (key) => {
    const spec = MATERIAL_BY_KEY[key];
    setForm((f) => ({
      ...f,
      material: key,
      sizes: spec?.size ? [{ w: spec.size.w, h: spec.size.h }] : f.sizes,
    }));
  };

  const toggleSurface = (key) => set({
    surfaces: form.surfaces.includes(key)
      ? form.surfaces.filter((s) => s !== key)
      : [...form.surfaces, key],
  });

  const toggleSize = (sz) => {
    const has = form.sizes.some((s) => s.w === sz.w && s.h === sz.h);
    set({
      sizes: has
        ? form.sizes.filter((s) => !(s.w === sz.w && s.h === sz.h))
        : [...form.sizes, { w: sz.w, h: sz.h }],
    });
  };

  async function save() {
    if (!form.name.trim()) return toast('A name is required.', 'error');
    if (isNew && needsFace && !files.length) return toast('Upload at least one tile face.', 'error');
    if (!form.sizes.length && needsFace) return toast('Pick at least one size.', 'error');
    setBusy(true);
    try {
      const saved = isNew
        ? await api.createProduct(form, files)
        : await api.updateProduct(product.id, form, files);
      toast('Product saved.', 'ok');
      onSaved(saved);
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title={isNew ? 'New product' : `Edit ${product.name}`}
      onClose={onClose}
      width={620}
      footer={
        <>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={save} disabled={busy}>
            {busy ? 'Saving…' : 'Save product'}
          </button>
        </>
      }
    >
      <div className="grid-2">
        <div className="field">
          <label>Name</label>
          <input className="input" value={form.name} onChange={(e) => set({ name: e.target.value })} />
        </div>
        <div className="field">
          <label>SKU</label>
          <input className="input" value={form.sku} onChange={(e) => set({ sku: e.target.value })} />
        </div>
        <div className="field">
          <label>Category</label>
          <select className="select" value={form.category} onChange={(e) => set({ category: e.target.value })}>
            <option value="">None</option>
            {categories.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Material</label>
          <select className="select" value={form.material} onChange={(e) => setMaterial(e.target.value)}>
            {MATERIALS.map((m) => (
              <option key={m.key} value={m.key}>{m.name}</option>
            ))}
          </select>
          <span className="tiny dim">
            {model === 'module' && 'Laid as discrete units with joints between them.'}
            {model === 'sheet' && 'One continuous pattern repeating over the surface — no joints.'}
            {model === 'solid' && 'A flat colour with a sheen. No image needed.'}
            {model === 'piece' && 'A single bounded rectangle placed on the floor.'}
            {model === 'joint' && 'Recolours the joints of the surface already in the photo.'}
          </span>
        </div>
      </div>

      {model === 'solid' && (
        <Section title="Paint colour">
          <div className="paint-grid">
            {PAINT_COLORS.map((c) => (
              <button
                key={c.hex}
                className={`paint-chip ${form.color?.toLowerCase() === c.hex ? 'active' : ''}`}
                style={{ background: c.hex }}
                title={c.name}
                onClick={() => set({ color: c.hex, name: form.name || c.name })}
              >
                <span>{c.name}</span>
              </button>
            ))}
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <input type="color" value={form.color} onChange={(e) => set({ color: e.target.value })}
                   style={{ width: 44, height: 34, border: 'none', background: 'none' }} />
            <input className="input grow" value={form.color} onChange={(e) => set({ color: e.target.value })} />
          </div>
          <span className="tiny dim">
            The swatch shown in the catalogue is generated from this. The
            renderer takes the colour itself, so changing it here changes the
            wall exactly, with no re-upload.
          </span>
        </Section>
      )}

      <Section title="Finish">
        <div className="row" style={{ marginBottom: 10 }}>
          {['matt', 'honed', 'satin', 'gloss', 'polished'].map((f) => (
            <button
              key={f}
              className={`opt grow ${form.finish === f ? 'active' : ''}`}
              onClick={() => set({ finish: f })}
            >
              {f}
            </button>
          ))}
        </div>
        <Slider
          label="Reflectivity"
          value={form.gloss}
          min={0}
          max={1.2}
          step={0.02}
          onChange={(v) => set({ gloss: v })}
          format={(v) => `${Math.round(v * 100)}%`}
        />
      </Section>

      <Section title="Can be applied to">
        <div className="filters">
          {SURFACE_TYPES.map((s) => (
            <button
              key={s.key}
              className={`chip ${form.surfaces.includes(s.key) ? 'active' : ''}`}
              onClick={() => toggleSurface(s.key)}
            >
              {s.name}
            </button>
          ))}
        </div>
      </Section>

      <Section title={model === 'sheet' ? 'Pattern repeat (mm)'
        : model === 'piece' ? 'Rug size (mm)' : 'Available sizes (mm)'}
        right={model === 'solid' ? <span className="tiny dim">not used for paint</span> : null}>
        <div className="size-grid">
          {TILE_SIZES.map((sz) => (
            <button
              key={`${sz.w}x${sz.h}`}
              className={`opt ${form.sizes.some((s) => s.w === sz.w && s.h === sz.h) ? 'active' : ''}`}
              onClick={() => toggleSize(sz)}
            >
              {sz.w} × {sz.h}
            </button>
          ))}
        </div>
      </Section>

      <Section title="Commercial">
        <div className="grid-2">
          <div className="field">
            <label>Price</label>
            <div className="row">
              <input className="input" type="number" step="0.01" min="0" value={form.price}
                     onChange={(e) => set({ price: e.target.value })} placeholder="e.g. 42.50" />
              <select className="select" style={{ width: 112 }} value={form.priceUnit}
                      onChange={(e) => set({ priceUnit: e.target.value })}>
                <option value="sqm">per m2</option>
                <option value="sqft">per sq ft</option>
                <option value="box">per box</option>
                <option value="pc">per piece</option>
              </select>
            </div>
          </div>
          <div className="field">
            <label>Pieces per box</label>
            <input className="input" type="number" min="1" value={form.piecesPerBox}
                   onChange={(e) => set({ piecesPerBox: e.target.value })} />
          </div>
        </div>
        <div className="field">
          <label>Coverage per box (m2)</label>
          <input className="input" type="number" step="0.01" min="0" value={form.coverageSqm}
                 onChange={(e) => set({ coverageSqm: e.target.value })} />
          <span className="tiny dim">
            Used by the quantity calculator to round an area up to whole boxes.
          </span>
        </div>
        <div className="field">
          <label>Description</label>
          <textarea className="textarea" value={form.description}
                    onChange={(e) => set({ description: e.target.value })} />
        </div>
      </Section>

      <Section title={isNew ? 'Product images' : 'Add more images'}>
        <input
          type="file"
          accept="image/*"
          multiple={model === 'module'}
          onChange={(e) => setFiles([...(e.target.files ?? [])])}
        />
        <p className="tiny dim" style={{ marginTop: 6 }}>
          {model === 'module' && (
            <>
              Upload 3–6 faces of the same design. The renderer picks one per
              unit at random, which is what stops a laid floor from visibly
              repeating.
            </>
          )}
          {model === 'sheet' && 'One tile-able image of a single pattern repeat. It is wrapped continuously, so its edges must match.'}
          {model === 'solid' && 'Nothing to upload — the swatch is generated from the colour above.'}
          {model === 'piece' && 'One image of the whole rug, cropped to its outline.'}
          {model === 'joint' && 'One image is enough; only the joint colour is drawn from it.'}
          {!isNew && ` Currently ${product.faces?.length ?? 0}.`}
        </p>
      </Section>
    </Modal>
  );
}

/* ----------------------------------------------------------------- rooms -- */

function RoomsTab() {
  const toast = useToast();
  const navigate = useNavigate();
  const categories = useSelector((s) => s.catalog.roomCategories);
  const [rooms, setRooms] = useState([]);
  const fileRef = useRef(null);
  const modelRef = useRef(null);
  const [busy, setBusy] = useState(false);

  const reload = () => api.rooms({ includeCustom: '1' }).then(setRooms).catch((e) => toast(e.message, 'error'));
  useEffect(() => { reload(); }, []);

  async function onUpload(e) {
    const files = [...(e.target.files ?? [])];
    e.target.value = '';
    for (const file of files) {
      try {
        await api.uploadRoom(file, { name: file.name.replace(/\.[^.]+$/, ''), isCustom: '0' });
      } catch (err) {
        toast(err.message, 'error');
      }
    }
    toast(`${files.length} room photo(s) added. Mark their surfaces in the Studio.`, 'ok');
    reload();
  }

  /**
   * A glTF room. Its geometry is already exact, so there is nothing to detect
   * -- it goes straight to the mesh-tagging Studio instead.
   */
  async function onModel(e) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setBusy(true);
    try {
      const room = await api.uploadModel(file, { name: file.name.replace(/\.[^.]+$/, '') });
      toast('Model uploaded. Tag its surfaces next.', 'ok');
      navigate(`/studio3d/${room.id}`);
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      setBusy(false);
    }
  }

  async function onDelete(room) {
    if (!confirm(`Delete "${room.name}"?`)) return;
    await api.deleteRoom(room.id);
    toast('Room deleted.', 'ok');
    reload();
  }

  async function setCategory(room, category) {
    await api.saveRoom(room.id, { category });
    reload();
  }

  return (
    <>
      <div className="row" style={{ marginBottom: 16 }}>
        <div className="grow" />
        <button className="btn" onClick={() => modelRef.current?.click()} disabled={busy}>
          <IconCube /> {busy ? 'Uploading…' : 'Add a 3D room'}
        </button>
        <input ref={modelRef} type="file" accept=".glb,.gltf,model/gltf-binary,model/gltf+json"
               hidden onChange={onModel} />
        <button className="btn btn-primary" onClick={() => fileRef.current?.click()}>
          <IconUpload /> Add room photos
        </button>
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={onUpload} />
      </div>

      {rooms.length ? (
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 76 }} />
              <th>Name</th>
              <th>Category</th>
              <th>Size</th>
              <th>Surfaces</th>
              <th>Source</th>
              <th style={{ width: 150 }} />
            </tr>
          </thead>
          <tbody>
            {rooms.map((r) => (
              <tr key={r.id}>
                <td><img src={r.thumb} alt="" style={{ width: 60, height: 40 }} /></td>
                <td><strong>{r.name}</strong></td>
                <td>
                  <select
                    className="select"
                    style={{ width: 150 }}
                    value={r.category ?? ''}
                    onChange={(e) => setCategory(r, e.target.value || null)}
                  >
                    <option value="">Uncategorised</option>
                    {categories.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                </td>
                <td className="muted tiny">
                  {r.kind === '3d' ? 'glTF model' : `${r.width}×${r.height}`}
                </td>
                <td className={r.objectList.length ? 'muted' : ''} style={r.objectList.length ? undefined : { color: 'var(--warn)' }}>
                  {r.objectList.length || 'none'}
                </td>
                <td className="muted tiny">{r.isCustom ? 'Visitor upload' : 'Preset'}</td>
                <td>
                  <div className="row" style={{ gap: 4 }}>
                    <button
                      className="btn btn-sm"
                      onClick={() => navigate(r.kind === '3d' ? `/studio3d/${r.id}` : `/studio/${r.id}`)}
                      disabled={r.kind === '360'}
                      title={r.kind === '360' ? 'Panorama planes are set in the room data' : undefined}
                    >
                      Surfaces
                    </button>
                    <button className="btn btn-ghost btn-icon btn-sm" onClick={() => onDelete(r)}>
                      <IconTrash />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty icon={<IconRoom />} title="No rooms yet" hint="Upload room photographs to get started." />
      )}
    </>
  );
}

/* ------------------------------------------------------------ categories -- */

function CategoriesTab() {
  const dispatch = useDispatch();
  const toast = useToast();
  const roomCategories = useSelector((s) => s.catalog.roomCategories);
  const productCategories = useSelector((s) => s.catalog.productCategories);
  const [roomName, setRoomName] = useState('');
  const [prodName, setProdName] = useState('');

  async function addRoomCat() {
    if (!roomName.trim()) return;
    await api.saveRoomCategory({ name: roomName.trim() });
    setRoomName('');
    dispatch(loadCatalog());
    toast('Category added.', 'ok');
  }

  async function addProdCat() {
    if (!prodName.trim()) return;
    await api.saveProductCategory({ name: prodName.trim() });
    setProdName('');
    dispatch(loadCatalog());
    toast('Category added.', 'ok');
  }

  return (
    <div className="grid-2" style={{ gap: 18 }}>
      <div className="card">
        <h3>Room categories</h3>
        <div className="row" style={{ marginBottom: 14 }}>
          <input className="input grow" placeholder="e.g. Hallway" value={roomName} onChange={(e) => setRoomName(e.target.value)} />
          <button className="btn btn-primary" onClick={addRoomCat}><IconPlus /></button>
        </div>
        {roomCategories.map((c) => (
          <div key={c.id} className="row-between" style={{ padding: '6px 0', borderBottom: '1px solid var(--line)' }}>
            <span>{c.name}</span>
            <span className="tiny dim">{c.count} rooms</span>
          </div>
        ))}
      </div>

      <div className="card">
        <h3>Product categories</h3>
        <div className="row" style={{ marginBottom: 14 }}>
          <input className="input grow" placeholder="e.g. Porcelain" value={prodName} onChange={(e) => setProdName(e.target.value)} />
          <button className="btn btn-primary" onClick={addProdCat}><IconPlus /></button>
        </div>
        {productCategories.map((c) => (
          <div key={c.id} className="row-between" style={{ padding: '6px 0', borderBottom: '1px solid var(--line)' }}>
            <span>{c.name}</span>
            <span className="tiny dim">{c.count} products</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- branding -- */

function BrandingTab() {
  const dispatch = useDispatch();
  const toast = useToast();
  const vendor = useSelector((s) => s.catalog.vendor);
  const [form, setForm] = useState(null);
  const logoRef = useRef(null);

  useEffect(() => { if (vendor && !form) setForm(structuredClone(vendor)); }, [vendor, form]);
  if (!form) return null;

  const setSetting = (k, v) => setForm((f) => ({ ...f, settings: { ...f.settings, [k]: v } }));

  async function save() {
    try {
      const saved = await api.saveVendor({
        name: form.name,
        primaryColor: form.primaryColor,
        settings: form.settings,
      });
      dispatch(setVendor(saved));
      toast('Branding saved.', 'ok');
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  async function onLogo(e) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    const saved = await api.uploadLogo(file);
    dispatch(setVendor(saved));
    setForm(structuredClone(saved));
    toast('Logo updated.', 'ok');
  }

  const embed = `<iframe src="${window.location.origin}/" width="100%" height="720" style="border:0" allowfullscreen></iframe>`;

  return (
    <>
      <div className="card">
        <h3>Identity</h3>
        <div className="grid-2">
          <div className="field">
            <label>Business name</label>
            <input className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </div>
          <div className="field">
            <label>Accent colour</label>
            <div className="row">
              <input
                type="color"
                value={form.primaryColor}
                onChange={(e) => setForm({ ...form, primaryColor: e.target.value })}
                style={{ width: 44, height: 34, border: 'none', background: 'none' }}
              />
              <input className="input grow" value={form.primaryColor} onChange={(e) => setForm({ ...form, primaryColor: e.target.value })} />
            </div>
          </div>
        </div>
        <div className="field">
          <label>Logo</label>
          <div className="row">
            {form.logo && <img src={form.logo} alt="" style={{ height: 34, borderRadius: 6 }} />}
            <button className="btn" onClick={() => logoRef.current?.click()}><IconUpload /> Upload logo</button>
            <input ref={logoRef} type="file" accept="image/*" hidden onChange={onLogo} />
          </div>
        </div>
        <div className="grid-2">
          <div className="field">
            <label>Currency symbol</label>
            <input className="input" value={form.settings.currencySymbol ?? '$'}
                   onChange={(e) => setSetting('currencySymbol', e.target.value)} />
          </div>
          <div className="field">
            <label>Default wastage %</label>
            <input className="input" type="number" min="0" max="40"
                   value={form.settings.wastagePercent ?? 10}
                   onChange={(e) => setSetting('wastagePercent', Number(e.target.value))} />
          </div>
        </div>
        <div className="grid-2">
          <div className="field">
            <label>Contact email</label>
            <input className="input" value={form.settings.contactEmail ?? ''}
                   onChange={(e) => setSetting('contactEmail', e.target.value)} />
          </div>
          <div className="field">
            <label>Contact phone</label>
            <input className="input" value={form.settings.contactPhone ?? ''}
                   onChange={(e) => setSetting('contactPhone', e.target.value)} />
          </div>
        </div>
        <div className="field">
          <label>Watermark on downloaded images</label>
          <input
            className="input"
            value={form.settings.watermark ?? ''}
            onChange={(e) => setSetting('watermark', e.target.value)}
            placeholder="Leave empty for no watermark"
          />
        </div>
      </div>

      <div className="card">
        <h3>Visitor features</h3>
        <Switch label="Let visitors upload their own room photo" checked={form.settings.allowUpload !== false} onChange={(v) => setSetting('allowUpload', v)} />
        <Switch label="Compare mode" checked={form.settings.allowCompare !== false} onChange={(v) => setSetting('allowCompare', v)} />
        <Switch label="Download rendered image" checked={form.settings.allowDownload !== false} onChange={(v) => setSetting('allowDownload', v)} />
        <Switch label="Share links and QR codes" checked={form.settings.allowShare !== false} onChange={(v) => setSetting('allowShare', v)} />
        <Switch label="Product enquiry form" checked={form.settings.allowInquiry !== false} onChange={(v) => setSetting('allowInquiry', v)} />
        <Switch label="Quantity calculator" checked={form.settings.allowCalculator !== false} onChange={(v) => setSetting('allowCalculator', v)} />
        <Switch label="Save scheme" checked={form.settings.allowSaveRoom !== false} onChange={(v) => setSetting('allowSaveRoom', v)} />
        <Switch label="Wishlist" checked={form.settings.allowWishlist !== false} onChange={(v) => setSetting('allowWishlist', v)} />
        <Switch label="Show prices" checked={form.settings.allowPrice !== false} onChange={(v) => setSetting('allowPrice', v)} />
        <Switch label="PDF export" checked={form.settings.allowPdf !== false} onChange={(v) => setSetting('allowPdf', v)} />
      </div>

      <div className="card">
        <h3>Showroom kiosk</h3>
        <p className="tiny muted">
          Point a screen on the shop floor at the kiosk link below. It hides the
          Studio and Admin, enlarges the touch targets, and clears whatever the
          last customer was doing after a period of no input.
        </p>
        <div className="grid-2">
          <div className="field">
            <label>Reset after (seconds of no input)</label>
            <input
              className="input"
              type="number"
              min="20"
              max="1800"
              value={form.settings.kioskIdleSeconds ?? 120}
              onChange={(e) => setSetting('kioskIdleSeconds', Number(e.target.value) || 120)}
            />
          </div>
          <div className="field">
            <label>Attract screen message</label>
            <input
              className="input"
              value={form.settings.kioskMessage ?? ''}
              placeholder="See it in your room"
              onChange={(e) => setSetting('kioskMessage', e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Website integration</h3>
        <p className="tiny muted">Drop this into any page to embed the visualizer.</p>
        <textarea className="textarea" readOnly value={embed} onFocus={(e) => e.target.select()} />
      </div>

      <LinksCard />

      <button className="btn btn-primary" onClick={save}>Save settings</button>
    </>
  );
}

/* ---------------------------------------------------------------- stores -- */

function StoresTab() {
  const toast = useToast();
  const [stores, setStores] = useState([]);
  const blank = { name: '', address: '', city: '', phone: '', email: '' };
  const [form, setForm] = useState(blank);

  const reload = () => api.stores().then(setStores).catch(() => {});
  useEffect(() => { reload(); }, []);

  async function save() {
    if (!form.name.trim()) return toast('A store name is required.', 'error');
    try {
      await api.saveStore(form);
      setForm(blank);
      reload();
      toast('Store saved.', 'ok');
    } catch (e) { toast(e.message, 'error'); }
  }

  return (
    <>
      <div className="card">
        <h3>Add a showroom</h3>
        <div className="grid-2">
          <div className="field">
            <label>Name</label>
            <input className="input" value={form.name}
                   onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </div>
          <div className="field">
            <label>City</label>
            <input className="input" value={form.city}
                   onChange={(e) => setForm({ ...form, city: e.target.value })} />
          </div>
        </div>
        <div className="field">
          <label>Address</label>
          <input className="input" value={form.address}
                 onChange={(e) => setForm({ ...form, address: e.target.value })} />
        </div>
        <div className="grid-2">
          <div className="field">
            <label>Phone</label>
            <input className="input" value={form.phone}
                   onChange={(e) => setForm({ ...form, phone: e.target.value })} />
          </div>
          <div className="field">
            <label>Email</label>
            <input className="input" value={form.email}
                   onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </div>
        </div>
        <button className="btn btn-primary" onClick={save}><IconPlus /> Add store</button>
      </div>

      {stores.length ? (
        <table className="table">
          <thead>
            <tr><th>Name</th><th>City</th><th>Address</th><th>Phone</th><th style={{ width: 50 }} /></tr>
          </thead>
          <tbody>
            {stores.map((s) => (
              <tr key={s.id}>
                <td><strong>{s.name}</strong></td>
                <td className="muted">{s.city ?? '\u2014'}</td>
                <td className="muted">{s.address ?? '\u2014'}</td>
                <td className="muted">{s.phone ?? '\u2014'}</td>
                <td>
                  <button
                    className="btn btn-ghost btn-icon btn-sm"
                    onClick={async () => { await api.deleteStore(s.id); reload(); }}
                  >
                    <IconTrash />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty icon={<IconStore />} title="No showrooms yet"
               hint="Visitors can be pointed at these from the visualizer." />
      )}
    </>
  );
}

/* ------------------------------------------------------------ links & QR -- */

/**
 * Direct links, with a QR code for each.
 *
 * A printed code on a sample board that opens the visualizer already on the
 * right room is the whole point of a QR feature -- a code that lands on the
 * home page and makes the customer hunt is barely better than the URL.
 */
function LinksCard() {
  const [rooms, setRooms] = useState([]);
  const [roomId, setRoomId] = useState('');
  const [kiosk, setKiosk] = useState(false);
  const toast = useToast();

  useEffect(() => { api.rooms({ includeCustom: '1' }).then(setRooms).catch(() => {}); }, []);

  const origin = window.location.origin;
  const url = `${origin}${roomId ? `/visualizer/${roomId}` : '/'}${kiosk ? '?kiosk=1' : ''}`;

  return (
    <div className="card">
      <h3>Links &amp; QR codes</h3>
      <p className="tiny muted">
        Print one of these on a sample board or a price list and it opens
        straight into the room — 2D or 360, whichever that room is.
      </p>
      <div className="row" style={{ alignItems: 'flex-start', gap: 18 }}>
        <QrCode text={url} size={150} />
        <div className="grow">
          <div className="field">
            <label>Opens at</label>
            <select className="select" value={roomId} onChange={(e) => setRoomId(e.target.value)}>
              <option value="">The room picker</option>
              {rooms.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}{r.kind === '360' ? ' (360°)' : ''}
                </option>
              ))}
            </select>
          </div>
          <Switch label="Kiosk mode" checked={kiosk} onChange={setKiosk} />
          <input className="input" readOnly value={url} onFocus={(e) => e.target.select()} />
          <button
            className="btn"
            style={{ marginTop: 8 }}
            onClick={() => { navigator.clipboard.writeText(url); toast('Link copied.', 'ok'); }}
          >
            Copy link
          </button>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- leads -- */

function LeadsTab() {
  const [leads, setLeads] = useState([]);
  useEffect(() => { api.leads().then(setLeads).catch(() => {}); }, []);

  if (!leads.length) {
    return <Empty title="No enquiries yet" hint="Enquiries visitors send from the visualizer appear here." />;
  }

  return (
    <table className="table">
      <thead>
        <tr><th>When</th><th>Name</th><th>Email</th><th>Phone</th><th>Message</th></tr>
      </thead>
      <tbody>
        {leads.map((l) => (
          <tr key={l.id}>
            <td className="muted tiny">{l.created_at}</td>
            <td>{l.name ?? '—'}</td>
            <td>{l.email ?? '—'}</td>
            <td>{l.phone ?? '—'}</td>
            <td className="muted">{l.message ?? '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
