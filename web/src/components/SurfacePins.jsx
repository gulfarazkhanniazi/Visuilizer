import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { setActiveSurface } from '../store/vizSlice.js';
import { maskAnchor } from '../engine/masks.js';
import { SURFACE_BY_KEY } from '../engine/layouts.js';
import { useT } from '../i18n/index.jsx';

/**
 * Markers showing where the detected surfaces are.
 *
 * Automatic detection is invisible: the room opens looking exactly like the
 * photograph, and nothing says that the floor and each wall are separately
 * selectable. People click the tile they want and then click the picture,
 * which does nothing they can see. A pin on each surface makes the map of the
 * room legible -- what was found, what is on it, and what can be tapped.
 *
 * The pin sits at the pole of inaccessibility of the mask rather than its
 * centroid, so it never lands on an edge, in a cut-out, or on top of the
 * neighbouring wall's pin.
 */
export default function SurfacePins({ room, hover, onHover }) {
  const dispatch = useDispatch();
  const active = useSelector((s) => s.viz.activeSurface);
  const frame = useSelector((s) => s.viz.activeFrame);
  const states = useSelector((s) => s.viz.frames[frame]);
  const products = useSelector((s) => s.catalog.products);
  const view = useSelector((s) => s.viz.view);
  const t = useT();

  const ref = useRef(null);
  const [box, setBox] = useState(null);

  // The overlay sits inside the stage, so it measures the same rectangle the
  // canvas is drawn into and needs no access to the renderer at all.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setBox({ w: r.width, h: r.height });
    };
    measure();
    const obs = new ResizeObserver(measure);
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Anchors are pure geometry, so they only change when the room does.
  const anchors = useMemo(() => {
    const out = {};
    for (const o of room?.objectList ?? []) {
      out[o.name] = maskAnchor(o.mask, room.width, room.height);
    }
    return out;
  }, [room]);

  if (!room || !box) return <div className="pin-layer" ref={ref} />;

  // Same aspect-fit the present pass applies, inverted.
  const photoAspect = room.width / room.height;
  const canvasAspect = box.w / box.h;
  const scale = photoAspect > canvasAspect
    ? { x: 1, y: canvasAspect / photoAspect }
    : { x: photoAspect / canvasAspect, y: 1 };

  const place = (px, py) => {
    const ux = px / room.width;
    const uy = 1 - py / room.height;
    const vx = (ux - view.x) * scale.x * view.zoom + 0.5;
    const vy = (uy - view.y) * scale.y * view.zoom + 0.5;
    return { left: vx * 100, top: (1 - vy) * 100, on: vx > 0.02 && vx < 0.98 && vy > 0.02 && vy < 0.98 };
  };

  return (
    <div className="pin-layer" ref={ref}>
      {(room.objectList ?? []).map((o) => {
        const a = anchors[o.name];
        if (!a) return null;
        const pos = place(a.x, a.y);
        if (!pos.on) return null;

        const st = states?.[o.name];
        const product = products.find((p) => p.id === st?.productId);
        const isActive = active === o.name;
        const isHover = hover === o.name;

        return (
          <button
            key={o.name}
            type="button"
            className={`pin ${isActive ? 'active' : ''} ${isHover ? 'hover' : ''} ${st?.visible === false ? 'off' : ''}`}
            style={{ left: `${pos.left}%`, top: `${pos.top}%` }}
            onClick={(e) => { e.stopPropagation(); dispatch(setActiveSurface(o.name)); }}
            onPointerEnter={() => onHover?.(o.name)}
            onPointerLeave={() => onHover?.(null)}
            title={`${o.label ?? o.name} — ${product ? product.name : t('pins.nothing')}`}
          >
            <span className="pin-dot">
              {product
                ? <span className="pin-swatch" style={{ backgroundImage: `url(${product.thumb})` }} />
                : <span className="pin-plus">+</span>}
            </span>
            <span className="pin-label">
              <strong>{o.label ?? o.name}</strong>
              <span>
                {product
                  ? product.name
                  : t('pins.tapToTile', {
                    surface: (SURFACE_BY_KEY[o.product_surface]?.name ?? o.product_surface).toLowerCase(),
                  })}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * Show the pins by themselves for a moment when a room opens, then get out of
 * the way. Long enough to read, short enough not to be in the photograph.
 */
export function useIntroPins(roomId, enabled = true) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    if (!roomId || !enabled) return undefined;
    setShow(true);
    const t = setTimeout(() => setShow(false), 4200);
    return () => clearTimeout(t);
  }, [roomId, enabled]);
  return [show, setShow];
}
