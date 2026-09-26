import { useCallback, useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import Renderer3D from '../engine/Renderer3D.js';
import { Spinner } from './ui.jsx';
import { useT } from '../i18n/index.jsx';
import { IconZoomIn, IconZoomOut, IconReset } from './Icons.jsx';

/**
 * Viewer for a modelled room.
 *
 * The two photographic paths give the visitor a fixed viewpoint because a
 * photograph has one. A model does not, so this is the path where "walk round
 * it and look again" is actually possible -- orbit, dolly, and a reset for
 * when someone has spun themselves into a corner.
 */
export default function Stage3D({ room, rendererRef, framesOverride }) {
  const products = useSelector((s) => s.catalog.products);
  const storeFrames = useSelector((s) => s.viz.frames);
  const storeActive = useSelector((s) => s.viz.activeFrame);
  // The Studio previews a draft that is not in the store yet, so it passes its
  // own frames in rather than dispatching half-finished surfaces.
  const frames = framesOverride ?? storeFrames;
  const activeFrame = framesOverride ? 'left' : storeActive;
  const t = useT();

  const [canvas, setCanvas] = useState(null);
  const localRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [auto, setAuto] = useState(false);

  const attach = useCallback((el) => setCanvas(el), []);

  useEffect(() => {
    if (!canvas) return undefined;
    let r;
    try {
      r = new Renderer3D(canvas);
    } catch (e) {
      setError(e.message || t('common.noWebgl'));
      return undefined;
    }
    localRef.current = r;
    if (rendererRef) rendererRef.current = r;
    r.start();
    return () => {
      r.dispose();
      if (localRef.current === r) localRef.current = null;
      if (rendererRef?.current === r) rendererRef.current = null;
    };
  }, [canvas, rendererRef, t]);

  useEffect(() => {
    const r = localRef.current;
    if (!r || !room?.model) return undefined;
    let cancelled = false;
    setReady(false);
    setLoading(true);
    r.setRoom(room, room.model)
      .then(() => {
        if (cancelled) return;
        setReady(true);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [room, canvas]);

  useEffect(() => {
    const r = localRef.current;
    if (!r || !ready) return undefined;
    let cancelled = false;
    (async () => {
      const needed = new Set();
      for (const frame of Object.values(frames)) {
        for (const st of Object.values(frame)) if (st.productId) needed.add(st.productId);
      }
      await Promise.all([...needed].map((id) => {
        const p = products.find((x) => x.id === id);
        return p ? r.loadProduct(p).catch(() => null) : null;
      }));
      if (cancelled) return;
      for (const [frame, surfaces] of Object.entries(frames)) {
        for (const [name, st] of Object.entries(surfaces)) {
          r.applyState(frame, name, st, products.find((p) => p.id === st.productId));
        }
      }
      r.setFrame(activeFrame);
      r.invalidate();
    })();
    return () => { cancelled = true; };
  }, [frames, products, ready, activeFrame]);

  useEffect(() => {
    if (!canvas) return undefined;
    const obs = new ResizeObserver(() => localRef.current?.invalidate());
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [canvas]);

  return (
    <>
      <canvas ref={attach} style={{ touchAction: 'none', cursor: 'grab' }} />

      {(loading || !ready) && <Spinner label={t('common.preparing')} />}
      {error && <div className="loading-veil"><strong>{error}</strong></div>}

      <div className="stage-bottom">
        <button className="btn btn-ghost btn-icon" onClick={() => localRef.current?.zoom(0.22)} title={t('toolbar.zoomOut')}>
          <IconZoomOut />
        </button>
        <button className="btn btn-ghost btn-icon" onClick={() => localRef.current?.zoom(-0.22)} title={t('toolbar.zoomIn')}>
          <IconZoomIn />
        </button>
        <button className="btn btn-ghost btn-icon" onClick={() => localRef.current?.resetView()} title={t('toolbar.fit')}>
          <IconReset />
        </button>
        <button
          className={`btn btn-sm ${auto ? 'btn-primary' : 'btn-ghost'}`}
          onClick={() => {
            const next = !auto;
            setAuto(next);
            localRef.current?.setAutoRotate(next);
          }}
        >
          {t('pano.autoSpin')}
        </button>
      </div>
    </>
  );
}
