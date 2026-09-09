import { useEffect, useMemo } from 'react';
import { Routes, Route, NavLink, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { loadCatalog } from './store/catalogSlice.js';
import { resetSurfaces } from './store/vizSlice.js';
import { track } from './api/client.js';
import { isKiosk, useIdleReset } from './kiosk.js';
import { ToastProvider } from './components/ui.jsx';
import LanguagePicker, { I18nProvider, useT } from './i18n/index.jsx';
import RoomSelect from './pages/RoomSelect.jsx';
import Visualizer from './pages/Visualizer.jsx';
import Studio from './pages/Studio.jsx';
import Studio3D from './pages/Studio3D.jsx';
import Admin from './pages/Admin.jsx';

export default function App() {
  return (
    <I18nProvider>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </I18nProvider>
  );
}

function Shell() {
  const dispatch = useDispatch();
  const vendor = useSelector((s) => s.catalog.vendor);
  const status = useSelector((s) => s.catalog.status);
  const location = useLocation();
  const navigate = useNavigate();
  const t = useT();

  const kiosk = useMemo(() => isKiosk(), []);

  useEffect(() => {
    if (status === 'idle') dispatch(loadCatalog());
  }, [status, dispatch]);

  // One per browser tab, which is what makes the funnel countable in sessions.
  useEffect(() => { track('session_start'); }, []);

  useEffect(() => {
    if (vendor?.primaryColor) {
      document.documentElement.style.setProperty('--accent', vendor.primaryColor);
    }
    if (vendor?.name) document.title = `${vendor.name} · Visualizer`;
  }, [vendor]);

  useEffect(() => {
    document.documentElement.classList.toggle('kiosk', kiosk);
  }, [kiosk]);

  /**
   * Put the kiosk back to the room list when the customer walks away, and drop
   * whatever they had applied -- the next person should not inherit it.
   */
  const idleSeconds = kiosk ? (vendor?.settings?.kioskIdleSeconds ?? 120) : 0;
  const idle = useIdleReset(idleSeconds, () => {
    if (location.pathname === '/') return;
    dispatch(resetSurfaces());
    track('kiosk_reset');
    navigate('/', { replace: true });
  }, kiosk);

  const immersive = /^\/(visualizer|studio|s)\//.test(location.pathname);

  return (
    <div className={`app ${kiosk ? 'is-kiosk' : ''}`}>
      <header className="topbar">
        <NavLink to="/" className="brand">
          {vendor?.logo
            ? <img src={vendor.logo} alt={vendor.name} />
            : <span className="brand-mark">◪</span>}
          <span className="desktop-only">{vendor?.name ?? 'Surface Visualizer'}</span>
        </NavLink>

        {!kiosk && (
          <nav className="topnav">
            <NavLink to="/" end>{t('nav.rooms')}</NavLink>
            <NavLink to="/studio">{t('nav.studio')}</NavLink>
            <NavLink to="/admin">{t('nav.admin')}</NavLink>
          </nav>
        )}

        <div className="spacer" />
        <LanguagePicker />
        {status === 'error' && (
          <span className="tiny" style={{ color: 'var(--danger)' }}>{t('app.offline')}</span>
        )}
      </header>

      <Routes>
        <Route path="/" element={<RoomSelect />} />
        <Route path="/visualizer/:roomId" element={<Visualizer />} />
        <Route path="/s/:shareCode" element={<Visualizer />} />
        {!kiosk && <Route path="/studio" element={<Studio />} />}
        {!kiosk && <Route path="/studio/:roomId" element={<Studio />} />}
        {!kiosk && <Route path="/studio3d/:roomId" element={<Studio3D />} />}
        {!kiosk && <Route path="/admin" element={<Admin />} />}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      {kiosk && idle && location.pathname === '/' && (
        <div className="attract" onPointerDown={(e) => e.currentTarget.remove()}>
          <div>
            {vendor?.logo && <img src={vendor.logo} alt="" />}
            <h1>{vendor?.settings?.kioskMessage || t('kiosk.attract')}</h1>
            <p>{t('kiosk.tap')}</p>
          </div>
        </div>
      )}
      {immersive && null}
    </div>
  );
}
