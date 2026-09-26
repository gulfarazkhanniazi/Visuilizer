import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client.js';
import { Empty, useToast } from '../components/ui.jsx';
import { IconChart } from '../components/Icons.jsx';
import { SURFACE_BY_KEY } from '../engine/layouts.js';

/**
 * Usage dashboard.
 *
 * The question a tile merchant actually has is not "how many hits" -- it is
 * "which of my products do people put in their rooms, and how many of them get
 * far enough to ask a price". So the funnel and the product ranking lead, and
 * traffic is one line behind them.
 *
 * Charts are inline SVG against the app's own tokens. Two series means two
 * hues, taken in fixed slot order and validated against this surface for
 * colour-vision separation; everything single-series is one hue, because the
 * bar length is already carrying the magnitude and a second colour would be
 * decoration pretending to be data.
 */

const RANGES = [
  { days: 7, label: '7 days' },
  { days: 30, label: '30 days' },
  { days: 90, label: '90 days' },
  { days: 365, label: '12 months' },
];

/** 1,284 / 12.9K / 1.4M -- a dashboard number, not an accounting one. */
function compact(n) {
  const v = Number(n) || 0;
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 10_000) return `${(v / 1000).toFixed(1)}K`;
  return v.toLocaleString();
}

const shortDay = (iso) => {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
};

export default function AnalyticsTab() {
  const toast = useToast();
  const [days, setDays] = useState(30);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(true);
  const [table, setTable] = useState(false);

  useEffect(() => {
    let alive = true;
    setBusy(true);
    api.analytics(days)
      .then((d) => alive && setData(d))
      .catch((e) => alive && toast(e.message, 'error'))
      .finally(() => alive && setBusy(false));
    return () => { alive = false; };
  }, [days, toast]);

  if (busy && !data) return <Empty icon={<IconChart />} title="Loading…" />;
  if (!data) return null;

  const t = data.totals;
  const nothing = t.events === 0;

  return (
    <>
      <div className="row" style={{ marginBottom: 18 }}>
        <div className="filters" style={{ margin: 0 }}>
          {RANGES.map((r) => (
            <button
              key={r.days}
              className={`chip ${days === r.days ? 'active' : ''}`}
              onClick={() => setDays(r.days)}
            >
              {r.label}
            </button>
          ))}
        </div>
        <div className="grow" />
        <button className={`chip ${table ? 'active' : ''}`} onClick={() => setTable((v) => !v)}>
          {table ? 'Charts' : 'Table'}
        </button>
      </div>

      {nothing ? (
        <Empty
          icon={<IconChart />}
          title="Nothing recorded yet"
          hint="Visits, the products people apply and the enquiries they send will appear here once the visualizer is being used."
        />
      ) : (
        <>
          <div className="viz-hero card">
            <div>
              <span className="viz-label">Sessions</span>
              <div className="viz-figure">{compact(t.sessions)}</div>
              <span className="tiny dim">
                {compact(t.visitors)} browsers · {compact(t.events)} recorded actions
                {' '}over the last {data.days} days
              </span>
            </div>
            <FunnelChart steps={data.funnel} />
          </div>

          <div className="viz-tiles">
            <StatTile label="Rooms opened" value={t.roomViews} />
            <StatTile label="Products applied" value={t.applies} />
            <StatTile label="Own photos uploaded" value={t.uploads} />
            <StatTile label="Quantities priced" value={t.calculators} />
            <StatTile label="Schemes kept" value={t.saves + t.downloads + t.shares}
                      note="saved, downloaded or shared" />
            <StatTile label="Enquiries" value={t.leads} accent />
          </div>

          {table ? (
            <DailyTable daily={data.daily} />
          ) : (
            <div className="card">
              <div className="row-between" style={{ marginBottom: 4 }}>
                <h3>Activity per day</h3>
                <Legend items={[['Sessions', 'var(--series-1)'], ['Products applied', 'var(--series-2)']]} />
              </div>
              <TimeSeries daily={data.daily} />
            </div>
          )}

          <div className="grid-2" style={{ gap: 18, alignItems: 'start' }}>
            <div className="card">
              <h3>Most applied products</h3>
              <p className="tiny dim" style={{ marginBottom: 12 }}>
                What people actually put on their walls and floors — not what they
                clicked past.
              </p>
              <RankedBars
                rows={data.topProducts.map((r) => ({
                  id: r.id,
                  label: r.name ?? 'Deleted product',
                  sub: r.material,
                  thumb: r.thumb,
                  value: r.n,
                  note: `${r.sessions} session${r.sessions === 1 ? '' : 's'}`,
                }))}
                empty="No product has been applied yet."
              />
            </div>

            <div className="card">
              <h3>Most opened rooms</h3>
              <p className="tiny dim" style={{ marginBottom: 12 }}>
                Which of your room photographs earn their place.
              </p>
              <RankedBars
                rows={data.topRooms.map((r) => ({
                  id: r.id,
                  label: r.name ?? 'Deleted room',
                  thumb: r.thumb,
                  value: r.n,
                }))}
                empty="No room has been opened yet."
              />

              <h3 style={{ marginTop: 22 }}>Surfaces people change</h3>
              <RankedBars
                rows={data.topSurfaces.map((r) => ({
                  id: r.id,
                  label: SURFACE_BY_KEY[r.id]?.name ?? r.id,
                  value: r.n,
                }))}
                empty="Nothing applied yet."
              />
            </div>
          </div>

          <p className="tiny dim" style={{ marginTop: 18 }}>
            Counted against the anonymous browser id the wishlist already uses.
            No names, no addresses and no cross-site tracking — a session is one
            browser tab.
          </p>
        </>
      )}
    </>
  );
}

/* ----------------------------------------------------------------- tiles -- */

function StatTile({ label, value, note, accent }) {
  return (
    <div className={`viz-tile ${accent ? 'accent' : ''}`}>
      <span className="viz-label">{label}</span>
      <strong>{compact(value)}</strong>
      {note && <span className="tiny dim">{note}</span>}
    </div>
  );
}

function Legend({ items }) {
  return (
    <div className="viz-legend">
      {items.map(([label, color]) => (
        <span key={label}>
          <i style={{ background: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}

/* ----------------------------------------------------------- time series -- */

const W = 760;
const H = 210;
const PAD = { l: 40, r: 58, t: 14, b: 26 };

/**
 * Sessions and applications on one scale.
 *
 * Deliberately one y-axis for both: they are the same kind of count, and a
 * second axis would let any pair of lines be drawn crossing at will.
 */
function TimeSeries({ daily }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);

  const max = Math.max(4, ...daily.map((d) => Math.max(d.sessions, d.applies)));
  const ticks = niceTicks(max, 4);
  const top = ticks[ticks.length - 1];

  const x = (i) => PAD.l + (i * (W - PAD.l - PAD.r)) / Math.max(1, daily.length - 1);
  const y = (v) => PAD.t + (1 - v / top) * (H - PAD.t - PAD.b);

  const path = (key) => daily.map((d, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(d[key]).toFixed(1)}`).join(' ');
  const last = daily[daily.length - 1] ?? { sessions: 0, applies: 0 };

  function onMove(e) {
    const r = ref.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (daily.length - 1));
    setHover(i >= 0 && i < daily.length ? i : null);
  }

  return (
    <div className="viz-chart">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Sessions and products applied, per day"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line className="viz-grid" x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} />
            <text className="viz-tick" x={PAD.l - 8} y={y(v) + 4} textAnchor="end">{compact(v)}</text>
          </g>
        ))}

        {daily.length > 1 && (
          <>
            <path className="viz-line" d={path('sessions')} stroke="var(--series-1)" />
            <path className="viz-line" d={path('applies')} stroke="var(--series-2)" />
          </>
        )}

        {/* End markers, ringed in the surface colour so they stay legible
            wherever the two lines cross. */}
        {daily.length > 0 && [['sessions', 'var(--series-1)'], ['applies', 'var(--series-2)']].map(([k, c]) => (
          <circle key={k} cx={x(daily.length - 1)} cy={y(last[k])} r="4.5" fill={c}
                  stroke="var(--bg-2)" strokeWidth="2" />
        ))}

        {/* Only the endpoints are labelled -- a number on every point is noise. */}
        <text className="viz-endlabel" x={x(daily.length - 1) + 10} y={y(last.sessions) + 4}>
          {compact(last.sessions)}
        </text>
        <text className="viz-endlabel" x={x(daily.length - 1) + 10} y={y(last.applies) + 4}>
          {compact(last.applies)}
        </text>

        <text className="viz-tick" x={PAD.l} y={H - 8}>{shortDay(daily[0]?.day ?? '')}</text>
        <text className="viz-tick" x={W - PAD.r} y={H - 8} textAnchor="end">
          {shortDay(daily[daily.length - 1]?.day ?? '')}
        </text>

        {hover != null && (
          <>
            <line className="viz-crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} />
            <circle cx={x(hover)} cy={y(daily[hover].sessions)} r="4.5" fill="var(--series-1)"
                    stroke="var(--bg-2)" strokeWidth="2" />
            <circle cx={x(hover)} cy={y(daily[hover].applies)} r="4.5" fill="var(--series-2)"
                    stroke="var(--bg-2)" strokeWidth="2" />
          </>
        )}
      </svg>

      {hover != null && (
        <div
          className="viz-tip"
          style={{
            left: `${(x(hover) / W) * 100}%`,
            transform: `translateX(${x(hover) > W * 0.6 ? '-102%' : '8px'})`,
          }}
        >
          <strong>{shortDay(daily[hover].day)}</strong>
          <span><i style={{ background: 'var(--series-1)' }} />Sessions <b>{daily[hover].sessions}</b></span>
          <span><i style={{ background: 'var(--series-2)' }} />Applied <b>{daily[hover].applies}</b></span>
        </div>
      )}
    </div>
  );
}

function niceTicks(max, count) {
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) * mag;
  const out = [];
  for (let v = 0; v <= max + step * 0.001; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

/* ---------------------------------------------------------------- funnel -- */

/**
 * Where people stop.
 *
 * One hue: the bar length is the magnitude, so colouring the stages differently
 * would be a second encoding of nothing. Counted in sessions, so somebody who
 * tried forty tiles is one person who got to step two.
 */
function FunnelChart({ steps }) {
  const top = Math.max(1, steps[0]?.n ?? 1);
  return (
    <div className="viz-funnel">
      {steps.map((s, i) => {
        const pct = (s.n / top) * 100;
        return (
          <div key={s.key} className="viz-funnel-row" title={`${s.n} of ${top} sessions`}>
            <span className="viz-label">{s.key}</span>
            <div className="viz-track">
              <div className="viz-fill" style={{ width: `${Math.max(pct, s.n ? 1.5 : 0)}%` }} />
            </div>
            <span className="viz-value">
              {compact(s.n)}
              {i > 0 && <em>{Math.round(pct)}%</em>}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* -------------------------------------------------------------- rankings -- */

function RankedBars({ rows, empty }) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  if (!rows.length) return <p className="tiny dim">{empty}</p>;

  return (
    <div className="viz-rank">
      {rows.map((r) => (
        <div key={r.id} className="viz-rank-row" title={`${r.label}: ${r.value}`}>
          {r.thumb
            ? <span className="viz-rank-thumb" style={{ backgroundImage: `url(${r.thumb})` }} />
            : <span className="viz-rank-thumb blank" />}
          <span className="viz-rank-name">
            <strong>{r.label}</strong>
            {r.sub && <span className="tiny dim">{r.sub}</span>}
          </span>
          <span className="viz-track">
            <span className="viz-fill" style={{ width: `${(r.value / max) * 100}%` }} />
          </span>
          <span className="viz-value">
            {compact(r.value)}
            {r.note && <em>{r.note}</em>}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ----------------------------------------------------------- table view -- */

function DailyTable({ daily }) {
  const rows = useMemo(() => [...daily].reverse(), [daily]);
  return (
    <div className="card">
      <h3>Activity per day</h3>
      <table className="table viz-table">
        <thead>
          <tr>
            <th>Day</th><th>Sessions</th><th>Rooms opened</th>
            <th>Products applied</th><th>Kept or asked</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.day}>
              <td>{d.day}</td>
              <td>{d.sessions}</td>
              <td>{d.views}</td>
              <td>{d.applies}</td>
              <td>{d.intents}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
