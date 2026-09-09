import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { api, visitorId, track } from '../api/client.js';
import { useToast, Empty } from '../components/ui.jsx';
import { useT } from '../i18n/index.jsx';
import { IconUpload, IconEdit, IconRoom, IconTrash } from '../components/Icons.jsx';

export default function RoomSelect() {
  const navigate = useNavigate();
  const toast = useToast();
  const t = useT();
  const categories = useSelector((s) => s.catalog.roomCategories);
  const vendor = useSelector((s) => s.catalog.vendor);

  const [rooms, setRooms] = useState([]);
  const [saved, setSaved] = useState([]);
  const [category, setCategory] = useState('all');
  const [busy, setBusy] = useState(null);   // null | 'uploading' | 'detecting'
  const fileRef = useRef(null);

  useEffect(() => {
    api.rooms({ category, owner: visitorId() })
      .then(setRooms)
      .catch((e) => toast(e.message, 'error'));
  }, [category, toast]);

  useEffect(() => {
    api.savedRooms().then(setSaved).catch(() => {});
  }, []);

  async function removeSaved(e, id) {
    e.stopPropagation();
    try {
      await api.deleteSavedRoom(id);
      setSaved((v) => v.filter((s) => s.id !== id));
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function onUpload(e) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setBusy('uploading');
    try {
      const room = await api.uploadRoom(file, {
        name: file.name.replace(/\.[^.]+$/, ''),
        owner: visitorId(),
        isCustom: '1',
      });

      // Find the floor and walls before handing the room over, so the visitor
      // sees a finished room rather than an empty canvas. If detection cannot
      // find anything we still continue -- the Studio is the fallback, not an
      // error state.
      track('room_upload', { roomId: room.id });
      setBusy('detecting');
      try {
        const res = await api.autoDetect(room.id);
        track('auto_detect', { roomId: room.id, meta: { surfaces: res.room.objectList.length } });
        const n = res.room.objectList.length;
        toast(n === 1 ? t('rooms.found1') : t('rooms.found', { n }), 'ok');
        navigate(`/visualizer/${room.id}`);
        return;
      } catch (detectErr) {
        // Say what actually went wrong; "could not find them" is unhelpful when
        // the real answer is that the detector never got to run.
        toast(t('rooms.detectFailed', { message: detectErr.message }), 'error', 7000);
        navigate(`/studio/${room.id}`);
        return;
      }
    } catch (err) {
      toast(err.message, 'error');
    } finally {
      setBusy(null);
    }
  }

  const allowUpload = vendor?.settings?.allowUpload !== false;

  return (
    <main className="page">
      <div className="page-inner">
        <div className="page-head">
          <h1>{t('rooms.title')}</h1>
          <p className="muted">{t('rooms.subtitle')}</p>
        </div>

        {saved.length > 0 && (
          <>
            <h3 style={{ fontSize: 15, marginBottom: 10 }}>{t('rooms.saved')}</h3>
            <div className="saved-strip">
              {saved.map((s) => (
                <div
                  key={s.id}
                  className="saved-card"
                  onClick={() => navigate(`/visualizer/${s.roomId}?saved=${s.id}`)}
                >
                  <span
                    className="thumb"
                    style={{ backgroundImage: `url(${s.preview ?? s.roomThumb})` }}
                  />
                  <span className="meta">
                    <strong>{s.name}</strong>
                    <span className="tiny muted">{s.roomName}</span>
                  </span>
                  <button className="del" onClick={(e) => removeSaved(e, s.id)} title={t('rooms.delete')}>
                    <IconTrash />
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

        <div className="filters">
          <button
            className={`chip ${category === 'all' ? 'active' : ''}`}
            onClick={() => setCategory('all')}
          >
            {t('rooms.all')}
          </button>
          {categories.map((c) => (
            <button
              key={c.id}
              className={`chip ${category === c.id ? 'active' : ''}`}
              onClick={() => setCategory(c.id)}
            >
              {c.name}
              {c.count > 0 && <span className="dim"> · {c.count}</span>}
            </button>
          ))}
        </div>

        <div className="room-grid">
          {allowUpload && (
            <>
              <button
                className="upload-card"
                onClick={() => fileRef.current?.click()}
                disabled={!!busy}
              >
                <IconUpload style={{ fontSize: 26 }} />
                <strong>
                  {busy === 'uploading' ? t('rooms.uploading')
                    : busy === 'detecting' ? t('rooms.detecting')
                      : t('rooms.upload')}
                </strong>
                <span className="tiny">
                  {busy === 'detecting' ? t('rooms.detectHint') : t('rooms.uploadHint')}
                </span>
              </button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                hidden
                onChange={onUpload}
              />
            </>
          )}

          {rooms.map((room) => (
            <div key={room.id} className="room-card">
              <div
                className="thumb"
                style={{ backgroundImage: `url(${room.thumb})` }}
                onClick={() => open(room, navigate)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && open(room, navigate)}
              >
                {room.kind === '360' && <span className="badge accent">360°</span>}
                {room.kind === '3d' && <span className="badge accent">3D</span>}
                {room.isCustom && <span className="badge accent" style={{ left: room.kind === '360' ? 62 : 9 }}>{t('rooms.yours')}</span>}
                {!room.objectList.length && <span className="badge">{t('rooms.needsSurfaces')}</span>}
              </div>
              <div className="meta row-between">
                <div className="grow" onClick={() => open(room, navigate)} style={{ cursor: 'pointer' }}>
                  <strong>{room.name}</strong>
                  <span className="tiny muted">
                    {room.kind === '360' ? t('rooms.panorama') : ''}
                    {room.kind === '3d' ? '3D · ' : ''}
                    {room.objectList.length
                      ? t('rooms.surfaceCount', { n: room.objectList.length })
                      : t('rooms.noSurfaces')}
                  </span>
                </div>
                <button
                  className="btn btn-ghost btn-icon"
                  title={t('rooms.editSurfaces')}
                  onClick={() => navigate(room.kind === '3d' ? `/studio3d/${room.id}` : `/studio/${room.id}`)}
                >
                  <IconEdit />
                </button>
              </div>
            </div>
          ))}
        </div>

        {!rooms.length && (
          <Empty
            icon={<IconRoom />}
            title={t('rooms.emptyTitle')}
            hint={t('rooms.emptyHint')}
          />
        )}
      </div>
    </main>
  );
}

/** A room with no authored surfaces has nothing to visualise -- send it to the studio. */
function open(room, navigate) {
  navigate(room.objectList.length ? `/visualizer/${room.id}` : `/studio/${room.id}`);
}
