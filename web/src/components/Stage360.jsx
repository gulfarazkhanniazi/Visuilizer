import { useCallback, useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import Renderer360 from '../engine/Renderer360.js';
import { Spinner, useToast } from './ui.jsx';
import { useT } from '../i18n/index.jsx';
import { IconReset, IconZoomIn, IconZoomOut } from './Icons.jsx';

/**
 * The 360 stage: a look-around view of a composited panorama.
 *
 * Shares all of the visualizer's panels, because those only ever touch redux
 * state -- only the renderer underneath is different.
 */
export default function Stage360({ room, rendererRef, onReady }) {
  const toast = useToast();
  const frames = useSelector((s) => s.viz.frames);
  const products = useSelector((s) => s.catalog.products);

  const [canvas, setCanvas] = useState(null);
  const canvasRef = useRef(null);
  const localRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fov, setFov] = useState(75);
  const [gyro, setGyro] = useState(false);
  const [auto, setAuto] = useState(false);
  const t = useT();

  const attach = useCallback((el) => {
    canvasRef.current = el;
    setCanvas(el);
  }, []);

  // --- create / destroy -----------------------------------------------------
  useEffect(() => {
    if (!canvas) return undefined;
    let r;
    try {
      r = new Renderer360(canvas);
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
  }, [canvas, rendererRef]);

  // --- load -----------------------------------------------------------------
  useEffect(() => {
    const r = localRef.current;
    if (!r || !room) return undefined;
    let cancelled = false;
    setReady(false);
    setLoading(true);
    r.setRoom(room, room.image)
      .then(() => {
        if (cancelled) return;
        if (room.settings?.blurRadius) r.setBlurRadius(room.settings.blurRadius);
        setReady(true);
        setLoading(false);
        onReady?.();
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [room, canvas, onReady]);

  // --- keep uniforms in step ------------------------------------------------
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
          const product = products.find((p) => p.id === st.productId);
          r.applyState(frame, name, st, product);
        }
      }
      r.invalidate();
    })();

    return () => { cancelled = true; };
  }, [frames, products, ready]);

  useEffect(() => {
    if (!canvas) return undefined;
    const obs = new ResizeObserver(() => localRef.current?.invalidate());
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [canvas]);

  // --- look around ----------------------------------------------------------
  const drag = useRef(null);

  const onPointerDown = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY };
    setAuto(false);
    localRef.current?.setAutoRotate(false);
  };

  const onPointerMove = (e) => {
    if (!drag.current) return;
    const r = localRef.current;
    r?.look(e.clientX - drag.current.x, e.clientY - drag.current.y);
    drag.current = { x: e.clientX, y: e.clientY };
  };

  const onPointerUp = () => { drag.current = null; };

  const zoom = (delta) => {
    const r = localRef.current;
    if (!r) return;
    r.setView({ fov: r.view.fov + delta });
    setFov(r.view.fov);
  };

  async function toggleGyro() {
    const r = localRef.current;
    if (!r) return;
    if (gyro) { r.disableGyro(); setGyro(false); return; }
    try {
      await r.enableGyro();
      setGyro(true);
      setAuto(false);
      r.setAutoRotate(false);
    } catch (e) {
      toast(e.message, 'error');
    }
  }

  return (
    <>
      <canvas
        ref={attach}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        onWheel={(e) => zoom(e.deltaY > 0 ? 4 : -4)}
        style={{ cursor: drag.current ? 'grabbing' : 'grab', touchAction: 'none' }}
      />

      {(loading || !ready) && <Spinner label={t('pano.building')} />}
      {error && <div className="loading-veil"><strong>{error}</strong></div>}

      <div className="stage-bottom">
        <button className="btn btn-ghost btn-icon" onClick={() => zoom(6)} title={t('toolbar.zoomOut')}>
          <IconZoomOut />
        </button>
        <span className="tiny muted" style={{ minWidth: 46, textAlign: 'center', alignSelf: 'center' }}>
          {Math.round(fov)}°
        </span>
        <button className="btn btn-ghost btn-icon" onClick={() => zoom(-6)} title={t('toolbar.zoomIn')}>
          <IconZoomIn />
        </button>
        <button
          className="btn btn-ghost btn-icon"
          onClick={() => {
            localRef.current?.setView({ yaw: 0, pitch: -0.22, fov: 75 });
            setFov(75);
          }}
          title={t('toolbar.fit')}
        >
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
        <button
          className={`btn btn-sm ${gyro ? 'btn-primary' : 'btn-ghost'}`}
          onClick={toggleGyro}
          title="Move your phone to look around"
        >
          Gyroscope
        </button>
      </div>
    </>
  );
}
