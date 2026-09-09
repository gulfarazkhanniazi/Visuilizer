import { useCallback, useEffect, useRef, useState } from 'react';
import Renderer from './Renderer.js';

/**
 * Owns the WebGL renderer for a canvas and keeps it in step with redux.
 *
 * The renderer is deliberately imperative: pushing uniforms is far cheaper than
 * rebuilding a scene graph, and a tile-size slider has to stay smooth while it
 * is being dragged.
 *
 * The canvas arrives through `attach`, a callback ref, rather than a ref object.
 * Callers only render the canvas once a room has loaded, so a plain ref would
 * still be null on the effect's single run and the renderer would never be
 * built; a callback ref re-runs the moment the element actually mounts.
 */
export default function useRenderer(room, { products, frames, compare, split, view, highlight }) {
  const [canvas, setCanvas] = useState(null);
  const canvasRef = useRef(null);
  const rendererRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const attach = useCallback((el) => {
    canvasRef.current = el;
    setCanvas(el);
  }, []);

  // --- create / destroy -----------------------------------------------------
  useEffect(() => {
    if (!canvas) return undefined;
    let renderer;
    try {
      renderer = new Renderer(canvas);
    } catch (e) {
      setError(e.message || 'WebGL 2 is not available in this browser.');
      return undefined;
    }
    rendererRef.current = renderer;
    renderer.start();
    return () => {
      renderer.dispose();
      if (rendererRef.current === renderer) rendererRef.current = null;
    };
  }, [canvas]);

  // --- load the room --------------------------------------------------------
  useEffect(() => {
    const r = rendererRef.current;
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
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [room, canvas]);

  // --- keep uniforms in sync ------------------------------------------------
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || !ready) return undefined;
    let cancelled = false;

    (async () => {
      // Load every product referenced by either frame before touching uniforms,
      // so a surface never renders one product's texture at another's size.
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
    rendererRef.current?.setCompare(compare, split);
  }, [compare, split, ready]);

  useEffect(() => {
    rendererRef.current?.setView(view);
  }, [view, ready]);

  useEffect(() => {
    rendererRef.current?.setHighlight(highlight);
  }, [highlight, ready]);

  // Re-render when the canvas is resized by a layout change.
  useEffect(() => {
    if (!canvas) return undefined;
    const obs = new ResizeObserver(() => rendererRef.current?.invalidate());
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [canvas]);

  /**
   * Convert a pointer position on the canvas into photo pixel coordinates,
   * undoing the aspect-fit, zoom and pan the present pass applies.
   */
  const screenToPhoto = useCallback((clientX, clientY) => {
    const r = rendererRef.current;
    const el = canvasRef.current;
    if (!r || !el || !r.room) return null;

    const rect = el.getBoundingClientRect();
    const vx = (clientX - rect.left) / rect.width;
    const vy = 1 - (clientY - rect.top) / rect.height;   // GL uv has y up

    const scale = r.presentMat.uniforms.uScale.value;
    const zoom = r.view.zoom;
    const ux = (vx - 0.5) / scale.x / zoom + r.view.x;
    const uy = (vy - 0.5) / scale.y / zoom + r.view.y;
    if (ux < 0 || ux > 1 || uy < 0 || uy > 1) return null;

    return { x: ux * r.size.w, y: (1 - uy) * r.size.h, u: ux, v: uy };
  }, []);

  return { attach, canvasRef, renderer: rendererRef, ready, loading, error, screenToPhoto };
}
