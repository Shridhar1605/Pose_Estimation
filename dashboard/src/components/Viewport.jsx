import { useEffect, useState } from 'react'
import { ACTION_COLOR, ALERT_ACTIONS, STREAM_SLOTS } from '../palette'

function useNow() {
  const [now, setNow] = useState(new Date())
  useEffect(() => { const id = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(id) }, [])
  return now
}
const two = (n) => String(n).padStart(2, '0')

function Feed({ slot, index, frame, expectedSource, now }) {
  const hasFrame = !!frame?.frame
  const alerts = hasFrame ? Object.entries(frame.actions || {}).filter(([a, n]) => ALERT_ACTIONS.has(a) && n > 0) : []
  const stamp = `${now.getFullYear()}-${two(now.getMonth() + 1)}-${two(now.getDate())} ${two(now.getHours())}:${two(now.getMinutes())}:${two(now.getSeconds())}`
  const progress = hasFrame && frame.total_frames ? Math.round((frame.source_frame / frame.total_frames) * 100) : null
  return (
    <div className={`feed ${alerts.length ? 'alert' : ''}`}>
      {hasFrame ? (
        <img src={`data:image/jpeg;base64,${frame.frame}`} alt={`${frame.source} annotated feed`} width={frame.width} height={frame.height} />
      ) : (
        <div className="nosignal">
          <div className="big">No signal</div>
          <div>CAM {two(index + 1)}</div>
          {expectedSource && <div className="sub">{expectedSource}</div>}
        </div>
      )}
      <div className="osd tl"><b>CAM {two(index + 1)}</b><span className="k">{hasFrame ? frame.source : ''}</span></div>
      <div className="osd tr">{hasFrame && <span className="dot rec" />}<span>{stamp}</span></div>
      {hasFrame && (
        <>
          <div className="osd bl">
            <span className="pill"><b>{frame.fps.toFixed(1)}</b> <span className="k">fps</span></span>
            <span className="pill"><b>{Math.round(frame.latency_ms)}</b> <span className="k">ms</span></span>
            <span className="pill"><b>{frame.person_count}</b> <span className="k">{frame.person_count === 1 ? 'person' : 'persons'}</span></span>
            {alerts.map(([a, n]) => <span key={a} className="pill action" style={{ background: ACTION_COLOR[a] }}>{a} ×{n}</span>)}
          </div>
          <div className="osd br">
            <span className="k">{frame.model.toUpperCase()} · {String(frame.device).replace('ExecutionProvider', '').toUpperCase()}</span>
            {progress !== null && <span className="k">{progress}%</span>}
          </div>
        </>
      )}
    </div>
  )
}

export default function Viewport({ mode, streams, selected, running }) {
  const now = useNow()
  const slots = mode === 'single' ? STREAM_SLOTS.slice(0, 1) : STREAM_SLOTS
  return (
    <main className="view">
      <div className={mode === 'single' ? 'grid-single' : 'grid-quad'}>
        {slots.map((slot, i) => (
          <Feed key={slot} slot={slot} index={i} frame={streams[slot]} now={now}
            expectedSource={running ? selected[i] : selected[i] ? `${selected[i].split('/').pop()} · press Start` : 'select a source'} />
        ))}
      </div>
    </main>
  )
}
