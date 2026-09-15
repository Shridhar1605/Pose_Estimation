import { ACTIONS, ACTION_COLOR, STREAM_SLOTS } from '../palette'

function Spark({ data, width = 150, height = 26 }) {
  if (!data || data.length < 2) return <svg width={width} height={height} />
  const xs = data.map((_, i) => (i / (data.length - 1)) * width)
  const vals = data.map((d) => d.fps)
  const max = Math.max(1, ...vals)
  const pts = xs.map((x, i) => `${x.toFixed(1)},${(height - 2 - (vals[i] / max) * (height - 4)).toFixed(1)}`).join(' ')
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      <polyline points={pts} fill="none" stroke="#F5A524" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

export default function Telemetry({ streams, fpsHistory, running }) {
  const live = STREAM_SLOTS.map((s) => streams[s]).filter(Boolean)
  const persons = live.reduce((a, f) => a + (f.person_count || 0), 0)
  const unique = live.reduce((a, f) => a + (f.unique_persons || 0), 0)
  const fps = live.reduce((a, f) => a + (f.fps || 0), 0)
  const latency = live.length ? live.reduce((a, f) => a + (f.latency_ms || 0), 0) / live.length : 0
  const mix = {}
  for (const f of live) for (const [a, n] of Object.entries(f.actions || {})) mix[a] = (mix[a] || 0) + n
  const mixTotal = Object.values(mix).reduce((a, b) => a + b, 0)
  const tracks = live.flatMap((f) => (f.tracks || []).map((t) => ({ ...t, stream: f.stream_id })))
    .sort((a, b) => (a.action === b.action ? a.id - b.id : ACTIONS.indexOf(b.action) - ACTIONS.indexOf(a.action)))

  return (
    <aside className="telem">
      <section>
        <div className="section-head"><span className="eyebrow">Now</span></div>
        <div className="kpis">
          <div className="kpi"><div className="v">{persons}</div><div className="l">persons in view</div></div>
          <div className="kpi"><div className="v">{unique}</div><div className="l">tracked ids</div></div>
          <div className="kpi"><div className="v">{fps.toFixed(1)}<small>fps</small></div><div className="l">pipeline output</div></div>
          <div className="kpi"><div className="v">{Math.round(latency)}<small>ms</small></div><div className="l">detect + pose</div></div>
        </div>
      </section>

      <section>
        <div className="section-head"><span className="eyebrow">Per-feed throughput</span><span className="meta">last 90 frames</span></div>
        {live.length === 0 && <div className="empty-note">{running ? 'Waiting for first frames…' : 'Start the pipeline to see throughput.'}</div>}
        {live.map((f) => (
          <div className="spark-row" key={f.stream_id}>
            <span className="lbl" title={`${f.source} · source ${f.native_fps} fps`}>CAM {String(STREAM_SLOTS.indexOf(f.stream_id) + 1).padStart(2, '0')}</span>
            <Spark data={fpsHistory[f.stream_id]} />
            <span className="val">{f.fps.toFixed(1)} <small>/ {Math.round(f.latency_ms)}ms</small></span>
          </div>
        ))}
      </section>

      <section>
        <div className="section-head"><span className="eyebrow">Action mix</span><span className="meta mono">{mixTotal} tracked</span></div>
        <div className="stack-bar" role="img" aria-label="Share of persons by action">
          {mixTotal === 0 ? <div style={{ flex: 1 }} /> : ACTIONS.filter((a) => mix[a]).map((a) => (
            <div key={a} style={{ flex: mix[a], background: ACTION_COLOR[a] }} title={`${a}: ${mix[a]}`} />
          ))}
        </div>
        <div className="legend">
          {ACTIONS.slice(0, 4).map((a) => (
            <div className="li" key={a}><span className="sw" style={{ background: ACTION_COLOR[a] }} />{a.toLowerCase()}<span className="n">{mix[a] || 0}</span></div>
          ))}
        </div>
      </section>

      <section>
        <div className="section-head"><span className="eyebrow">Tracks</span><span className="meta">{tracks.length ? `${Math.min(tracks.length, 14)} of ${tracks.length}` : ''}</span></div>
        {tracks.length === 0 ? <div className="empty-note">No active tracks.</div> : (
          <table className="tracks">
            <thead><tr><th>id</th><th>cam</th><th>action</th><th>conf</th><th>box</th></tr></thead>
            <tbody>
              {tracks.slice(0, 14).map((t) => (
                <tr key={`${t.stream}-${t.id}`}>
                  <td>#{t.id}</td>
                  <td>{String(STREAM_SLOTS.indexOf(t.stream) + 1).padStart(2, '0')}</td>
                  <td className="act"><span className="sw" style={{ background: ACTION_COLOR[t.action] }} />{t.action.toLowerCase()}</td>
                  <td>{t.conf.toFixed(2)}</td>
                  <td>{Math.round(t.bbox[2] - t.bbox[0])}×{Math.round(t.bbox[3] - t.bbox[1])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </aside>
  )
}
