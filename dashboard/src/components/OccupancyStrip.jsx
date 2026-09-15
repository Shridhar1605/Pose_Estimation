import { useEffect, useMemo, useRef, useState } from 'react'
import { ACTIONS, ACTION_COLOR } from '../palette'

const SECONDS = 120
const PAD = { l: 26, r: 8, t: 6, b: 16 }

// Rolling two-minute strip: one bar per second, stacked by action across all feeds.
export default function OccupancyStrip({ occupancy }) {
  const [hover, setHover] = useState(null)
  const box = useRef(null)
  const [size, setSize] = useState({ w: 1000, h: 130 })
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.max(200, e.contentRect.width), h: Math.max(80, e.contentRect.height) }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const W = size.w, H = size.h
  const innerW = W - PAD.l - PAD.r, innerH = H - PAD.t - PAD.b
  const slotW = innerW / SECONDS
  const max = Math.max(4, ...occupancy.map((o) => o.total))
  const ticks = useMemo(() => [0, Math.ceil(max / 2), max], [max])

  const bars = occupancy.map((o, i) => {
    const idx = SECONDS - occupancy.length + i
    const x = PAD.l + idx * slotW
    let y = PAD.t + innerH
    const segs = ACTIONS.filter((a) => o.byAction[a]).map((a) => {
      const h = (o.byAction[a] / max) * innerH
      y -= h
      return { a, y, h: Math.max(0, h - 1), n: o.byAction[a] }
    })
    return { x, segs, o, idx }
  })

  const onMove = (e) => {
    const r = box.current.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    const idx = Math.floor((px - PAD.l) / slotW)
    const b = bars.find((bb) => bb.idx === idx)
    setHover(b ? { b, left: e.clientX - r.left, top: e.clientY - r.top } : null)
  }

  return (
    <section className="occ" style={{ position: 'relative' }}>
      <div className="occ-head">
        <span className="eyebrow">Occupancy · last 2 min</span>
        <span className="hint">persons per second, stacked by action, all feeds</span>
      </div>
      <div ref={box} style={{ flex: 1, minHeight: 0, position: 'relative' }} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ display: 'block' }} role="img" aria-label="Persons per second over the last two minutes, stacked by action">
          {ticks.map((t) => {
            const y = PAD.t + innerH - (t / max) * innerH
            return (
              <g key={t}>
                <line x1={PAD.l} x2={W - PAD.r} y1={y} y2={y} stroke="#232D3A" strokeWidth="1" />
                <text x={PAD.l - 6} y={y + 3} textAnchor="end" fontSize="9" fill="#62708A" fontFamily="JetBrains Mono, monospace">{t}</text>
              </g>
            )
          })}
          {[0, 30, 60, 90, 120].map((s) => {
            const x = PAD.l + (SECONDS - s) * slotW
            return <text key={s} x={x} y={H - 4} textAnchor={s === 0 ? 'end' : s === 120 ? 'start' : 'middle'} fontSize="9" fill="#62708A" fontFamily="JetBrains Mono, monospace">{s === 0 ? 'now' : `-${s}s`}</text>
          })}
          {bars.map((b) => (
            <g key={b.idx}>
              {b.segs.map((s) => <rect key={s.a} x={b.x + 0.6} y={s.y} width={Math.max(0.5, slotW - 1.2)} height={s.h} fill={ACTION_COLOR[s.a]} rx="0.6" />)}
              {hover?.b.idx === b.idx && <rect x={b.x} y={PAD.t} width={slotW} height={innerH} fill="#F5A52418" />}
            </g>
          ))}
        </svg>
        {hover && (
          <div className="tip" style={{ left: Math.min(hover.left + 12, box.current.clientWidth - 150), top: Math.max(0, hover.top - 60) }}>
            <div><b>{hover.b.o.total}</b> persons · {Math.round((Date.now() - hover.b.o.t) / 1000)}s ago</div>
            {hover.b.segs.map((s) => <div key={s.a}><span className="sw" style={{ background: ACTION_COLOR[s.a] }} />{s.a.toLowerCase()}<span style={{ marginLeft: 'auto' }}>{s.n}</span></div>)}
          </div>
        )}
      </div>
    </section>
  )
}
