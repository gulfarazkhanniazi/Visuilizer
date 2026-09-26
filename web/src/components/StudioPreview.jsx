import { useEffect, useMemo, useState } from 'react';
import { useSelector } from 'react-redux';
import useRenderer from '../engine/useRenderer.js';
import { defaultSurfaceState } from '../engine/layouts.js';
import { Spinner } from './ui.jsx';

/**
 * Live preview inside the studio.
 *
 * Lays one test product on every surface so the author can check the
 * perspective and the mask edges against real tile lines -- the fastest way to
 * spot a quad that is a few pixels out.
 */
export default function StudioPreview({ room }) {
  const products = useSelector((s) => s.catalog.products);
  const [productId, setProductId] = useState(null);

  const testProduct = products.find((p) => p.id === productId) ?? products[0] ?? null;

  const frames = useMemo(() => {
    const build = () => Object.fromEntries((room.objectList ?? []).map((o) => [
      o.name,
      {
        ...defaultSurfaceState(o.product_surface),
        ...(o.defaults ?? {}),
        productId: testProduct?.id ?? null,
        grout: { size: 3, color: '#3f3f3d' },
      },
    ]));
    return { left: build(), right: {} };
  }, [room, testProduct]);

  const { attach, ready, loading } = useRenderer(room, {
    products,
    frames,
    compare: false,
    split: 0.5,
    view: { zoom: 1, x: 0.5, y: 0.5 },
    highlight: null,
  });

  useEffect(() => {
    if (!productId && products.length) setProductId(products[0].id);
  }, [products, productId]);

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={attach}
        style={{
          display: 'block',
          height: '86vh',
          width: 'auto',
          maxWidth: '100%',
          aspectRatio: `${room.width} / ${room.height}`,
        }}
      />
      {(loading || !ready) && <Spinner label="Rendering preview…" />}

      <div
        className="glass"
        style={{
          position: 'absolute', bottom: 10, right: 10,
          borderRadius: 8, padding: 6, display: 'flex', gap: 6, maxWidth: '70%',
          overflowX: 'auto',
        }}
      >
        {products.slice(0, 10).map((p) => (
          <button
            key={p.id}
            className={`tile-card ${p.id === testProduct?.id ? 'active' : ''}`}
            style={{ width: 40, flex: 'none' }}
            onClick={() => setProductId(p.id)}
            title={p.name}
          >
            <span className="swatch" style={{ backgroundImage: `url(${p.thumb})` }} />
          </button>
        ))}
      </div>
    </div>
  );
}
