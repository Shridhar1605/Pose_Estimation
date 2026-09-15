import { ACTION_COLOR, STREAM_SLOTS } from '../palette'

const two = (n) => String(n).padStart(2, '0')
const hhmmss = (ts) => { const d = new Date(ts); return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}` }

export default function EventLog({ events }) {
  return (
    <section className="log">
      <div className="section-head"><span className="eyebrow">Action changes</span><span className="meta">{events.length ? `${events.length} recent` : ''}</span></div>
      <div className="log-list" aria-live="polite">
        {events.length === 0 && <div className="empty-note">Posture changes and alerts (lying down, fighting) appear here.</div>}
        {events.map((e) => (
          <div key={e.key} className={`ev ${e.alert ? 'alert' : ''}`}>
            <span className="t">{hhmmss(e.ts)}</span>
            <span className="s">CAM {two(STREAM_SLOTS.indexOf(e.stream) + 1)} · #{e.id}</span>
            <span className="a"><span className="sw" style={{ background: ACTION_COLOR[e.to] }} />{e.from ? `${e.from.toLowerCase()} → ` : ''}<b>{e.to.toLowerCase()}</b></span>
          </div>
        ))}
      </div>
    </section>
  )
}
