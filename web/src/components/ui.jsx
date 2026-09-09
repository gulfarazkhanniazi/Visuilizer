import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { IconClose } from './Icons.jsx';

/* ------------------------------------------------------------------ toast -- */

const ToastCtx = createContext(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);

  const push = useCallback((message, kind = 'info', ms = 3200) => {
    const id = Math.random().toString(36).slice(2);
    setItems((v) => [...v, { id, message, kind }]);
    setTimeout(() => setItems((v) => v.filter((t) => t.id !== id)), ms);
  }, []);

  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toast-wrap">
        {items.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>{t.message}</div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/* ------------------------------------------------------------------ modal -- */

export function Modal({ title, children, footer, onClose, width }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose?.();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="modal-veil" onMouseDown={(e) => e.target === e.currentTarget && onClose?.()}>
      <div className="modal" style={width ? { width } : undefined}>
        <div className="modal-head">
          <h3>{title}</h3>
          <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Close">
            <IconClose />
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- inputs -- */

export function Switch({ checked, onChange, label }) {
  return (
    <div className="row-between" style={{ marginBottom: 10 }}>
      {label && <span className="tiny muted">{label}</span>}
      <button
        type="button"
        className={`switch ${checked ? 'on' : ''}`}
        onClick={() => onChange(!checked)}
        role="switch"
        aria-checked={checked}
        aria-label={label}
      />
    </div>
  );
}

export function Slider({ label, value, min, max, step = 1, unit = '', onChange, format }) {
  return (
    <div className="slider-row">
      <div className="row-between">
        <span className="muted">{label}</span>
        <span className="val">{format ? format(value) : `${value}${unit}`}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function Section({ title, children, right }) {
  return (
    <div className="section">
      {(title || right) && (
        <div className="row-between">
          <h4>{title}</h4>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function Empty({ icon, title, hint, action }) {
  return (
    <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--text-3)' }}>
      {icon && <div style={{ fontSize: 34, marginBottom: 10, opacity: 0.6 }}>{icon}</div>}
      <div style={{ fontWeight: 600, color: 'var(--text-2)', marginBottom: 5 }}>{title}</div>
      {hint && <div className="tiny">{hint}</div>}
      {action && <div style={{ marginTop: 14 }}>{action}</div>}
    </div>
  );
}

export function Spinner({ label }) {
  return (
    <div className="loading-veil">
      <div className="spinner" />
      {label && <div className="tiny">{label}</div>}
    </div>
  );
}

/**
 * Minimal QR encoder (byte mode, version chosen to fit, error level M).
 * Bundling a library for one 40-line matrix would be heavier than this is.
 */
export function QrCode({ text, size = 180 }) {
  const matrix = useQr(text);
  if (!matrix) return null;
  const n = matrix.length;
  const cell = size / (n + 8);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${n + 8} ${n + 8}`} style={{ background: '#fff', borderRadius: 8 }}>
      {matrix.map((row, y) => row.map((v, x) => (v ? (
        <rect key={`${x}-${y}`} x={x + 4} y={y + 4} width={1.02} height={1.02} fill="#000" />
      ) : null)))}
    </svg>
  );
}

function useQr(text) {
  const [m, setM] = useState(null);
  useEffect(() => {
    let alive = true;
    import('../lib/qr.js')
      .then((mod) => alive && setM(mod.encode(text)))
      .catch(() => alive && setM(null));
    return () => { alive = false; };
  }, [text]);
  return m;
}
