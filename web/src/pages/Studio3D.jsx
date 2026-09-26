import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { api } from '../api/client.js';
import { readModelMeshes } from '../engine/Renderer3D.js';
import { SURFACE_TYPES, defaultSurfaceState } from '../engine/layouts.js';
import Stage3D from '../components/Stage3D.jsx';
import { Section, useToast, Empty, Spinner } from '../components/ui.jsx';
import { IconCheck, IconChevron, IconCube } from '../components/Icons.jsx';
import AdminGate from './AdminAuth.jsx';

/**
 * Set up a modelled room.
 *
 * A photographed room needs its geometry solved; a modelled one already has
 * it, and the only thing missing is which of its meshes a customer is allowed
 * to re-tile. So this Studio is a list, not a canvas: tag the meshes, name
 * them, and preview a test tile on the result.
 *
 * Behind the same gate as the rest of the admin, because a glTF room is part
 * of the business's own library rather than something a visitor uploads.
 */
export default function Studio3D() {
  return <AdminGate><Studio3DPanel /></AdminGate>;
}

function Studio3DPanel() {
  const { roomId } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const products = useSelector((s) => s.catalog.products);

  const [room, setRoom] = useState(null);
  const [meshes, setMeshes] = useState(null);
  const [assign, setAssign] = useState({});     // mesh name -> { surface, label }
  const [testId, setTestId] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  // A ref, not a fresh object each render: Stage3D keys its renderer lifecycle
  // off this identity, so a new one every render would tear the scene down and
  // rebuild it on every keystroke.
  const rendererRef = useRef(null);

  useEffect(() => {
    api.room(roomId).then(setRoom).catch((e) => setError(e.message));
  }, [roomId]);

  // Read the mesh list straight out of the file, so the names offered are the
  // names the renderer will look for later.
  useEffect(() => {
    if (!room?.model) return;
    let alive = true;
    readModelMeshes(room.model)
      .then((list) => {
        if (!alive) return;
        setMeshes(list);
        const existing = Object.fromEntries(
          (room.objectList ?? []).map((o) => [o.name, { surface: o.product_surface, label: o.label }]),
        );
        setAssign(Object.fromEntries(list.map((m) => [
          m.name,
          existing[m.name] ?? (m.guess
            ? { surface: m.guess, label: prettyName(m.name) }
            : { surface: '', label: prettyName(m.name) }),
        ])));
      })
      .catch((e) => alive && setError(e.message));
    return () => { alive = false; };
  }, [room]);

  useEffect(() => {
    if (!testId && products.length) setTestId(products[0].id);
  }, [products, testId]);

  const surfaces = useMemo(() => Object.entries(assign)
    .filter(([, a]) => a.surface)
    .map(([name, a], i) => ({
      name,
      label: a.label || prettyName(name),
      product_surface: a.surface,
      meshes: [name],
      order: i,
      isMain: i === 0,
      defaults: a.surface === 'wall'
        ? { tileSize: { w: 300, h: 600 }, layout: 'brick', grout: { size: 2, color: '#f5f5f2' } }
        : { tileSize: { w: 600, h: 600 }, layout: 'grid', grout: { size: 2, color: '#c9c9c4' } },
    })), [assign]);

  const draftRoom = useMemo(
    () => (room ? { ...room, objectList: surfaces } : null),
    [room, surfaces],
  );

  const previewFrames = useMemo(() => ({
    left: Object.fromEntries(surfaces.map((o) => [
      o.name,
      { ...defaultSurfaceState(o.product_surface), ...o.defaults, productId: testId },
    ])),
    right: {},
  }), [surfaces, testId]);

  const set = (meshName, patch) => {
    setAssign((a) => ({ ...a, [meshName]: { ...a[meshName], ...patch } }));
    setDirty(true);
  };

  async function save() {
    setSaving(true);
    try {
      const thumb = rendererRef.current
        ? rendererRef.current.exportView(1).toDataURL('image/jpeg', 0.8)
        : undefined;
      const saved = await api.saveRoom(room.id, { objectList: surfaces, thumb });
      setRoom(saved);
      setDirty(false);
      toast('Room saved.', 'ok');
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      setSaving(false);
    }
  }

  if (error) {
    return (
      <main className="page">
        <div className="page-inner"><Empty icon={<IconCube />} title="Could not open the model" hint={error} /></div>
      </main>
    );
  }
  if (!room || !meshes) return <div className="stage"><Spinner label="Reading the model…" /></div>;

  const tagged = surfaces.length;

  return (
    <div className="studio">
      <aside className="panel">
        <div className="panel-head">
          <div>
            <h3>{room.name}</h3>
            <span className="tiny muted">{meshes.length} meshes · {tagged} tileable</span>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/admin')}>Admin</button>
        </div>

        <div className="panel-body">
          <Section title="Meshes">
            <p className="tiny dim" style={{ marginBottom: 10 }}>
              Tag the meshes a customer may re-tile. Anything left untagged keeps
              the material it came with — which is what the furniture in the room
              should do.
            </p>
            <div className="surface-list">
              {meshes.map((m) => (
                <div key={m.name} className={`mesh-row ${assign[m.name]?.surface ? 'on' : ''}`}>
                  <div className="row-between">
                    <strong className="tiny">{m.name}</strong>
                    <span className="tiny dim">
                      {m.size.x}×{m.size.y}×{m.size.z} m
                    </span>
                  </div>
                  <div className="row" style={{ marginTop: 6 }}>
                    <select
                      className="select"
                      style={{ width: 130 }}
                      value={assign[m.name]?.surface ?? ''}
                      onChange={(e) => set(m.name, { surface: e.target.value })}
                    >
                      <option value="">Not a surface</option>
                      {SURFACE_TYPES.map((s) => (
                        <option key={s.key} value={s.key}>{s.name}</option>
                      ))}
                    </select>
                    <input
                      className="input grow"
                      placeholder="Shown to the visitor"
                      value={assign[m.name]?.label ?? ''}
                      disabled={!assign[m.name]?.surface}
                      onChange={(e) => set(m.name, { label: e.target.value })}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Preview product">
            <select className="select" value={testId ?? ''} onChange={(e) => setTestId(e.target.value)}>
              {products.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Laid on every tagged mesh at real-world scale. If a 600 mm tile
              looks like a mosaic, the model was exported in the wrong units —
              re-export it in metres.
            </p>
          </Section>
        </div>

        <div className="panel-head" style={{ borderTop: '1px solid var(--line)', borderBottom: 'none' }}>
          <button className="btn btn-primary grow" onClick={save} disabled={!dirty || saving || !tagged}>
            <IconCheck /> {saving ? 'Saving…' : dirty ? 'Save room' : 'Saved'}
          </button>
          <button
            className="btn"
            onClick={() => navigate(`/visualizer/${room.id}`)}
            disabled={!tagged || dirty}
            title={dirty ? 'Save first' : 'Open in the visualizer'}
          >
            Open <IconChevron />
          </button>
        </div>
      </aside>

      <section className="stage">
        {draftRoom && (
          <Stage3D room={draftRoom} rendererRef={rendererRef} framesOverride={previewFrames} />
        )}
      </section>
    </div>
  );
}

/** "Wall_Back_01" -> "Wall Back 01", so the default label is already readable. */
function prettyName(name) {
  return String(name)
    .replace(/[_.-]+/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^./, (c) => c.toUpperCase());
}
