import { useCallback, useEffect, useRef, useState } from 'react'
import { ALERT_ACTIONS } from '../palette'

// Same-origin in production (FastAPI serves dist/); Vite proxies /api and /ws in dev.
const API = ''
const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/stream`

const FPS_HISTORY = 90      // samples per stream for the sparkline
const OCCUPANCY_SECONDS = 120
const MAX_EVENTS = 60

async function jsonFetch(url, opts) {
  const res = await fetch(API + url, opts)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.error || data.detail || `${res.status} ${res.statusText}`)
  return data
}

export function usePipeline() {
  const [system, setSystem] = useState(null)
  const [models, setModels] = useState([])
  const [videos, setVideos] = useState([])
  const [streams, setStreams] = useState({})
  const [fpsHistory, setFpsHistory] = useState({})
  const [occupancy, setOccupancy] = useState([])
  const [events, setEvents] = useState([])
  const [running, setRunning] = useState(false)
  const [runConfig, setRunConfig] = useState(null)
  const [startedAt, setStartedAt] = useState(null)
  const [connection, setConnection] = useState('connecting')
  const [error, setError] = useState(null)
  const [backendUp, setBackendUp] = useState(true)

  const wsRef = useRef(null)
  const latestRef = useRef({})
  const dirtyRef = useRef(false)
  const fpsRef = useRef({})
  const prevActionRef = useRef({})
  const reconnectRef = useRef({ timer: null, attempts: 0 })
  const occupancyRef = useRef([])

  // ── bootstrap ────────────────────────────────────────────────────────────
  const refreshVideos = useCallback(async () => {
    const d = await jsonFetch('/api/videos')
    setVideos(d.videos || [])
  }, [])

  useEffect(() => {
    (async () => {
      try {
        const [sys, mods, st] = await Promise.all([
          jsonFetch('/api/system'), jsonFetch('/api/models'), jsonFetch('/api/status'),
        ])
        setSystem(sys)
        setModels(mods.models || [])
        setBackendUp(true)
        if (st.running) {
          setRunning(true)
          setRunConfig(st.config)
          setStartedAt(Date.now() - (st.uptime_s || 0) * 1000)
        }
        await refreshVideos()
      } catch (e) {
        setBackendUp(false)
        setError('Backend is not reachable on :8000. Start it with `python server.py`.')
      }
    })()
  }, [refreshVideos])

  // ── websocket ────────────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= 1) return
    setConnection('connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    ws.onopen = () => { setConnection('connected'); reconnectRef.current.attempts = 0 }
    ws.onmessage = (ev) => {
      let msg
      try { msg = JSON.parse(ev.data) } catch { return }
      if (msg.type === 'frame') {
        latestRef.current[msg.stream_id] = msg
        dirtyRef.current = true
        const h = fpsRef.current[msg.stream_id] || (fpsRef.current[msg.stream_id] = [])
        h.push({ fps: msg.fps, latency: msg.latency_ms, t: msg.ts })
        if (h.length > FPS_HISTORY) h.shift()
        detectEvents(msg)
      } else if (msg.type === 'status') {
        if (msg.status === 'started') {
          setRunning(true)
          setStartedAt(Date.now())
          setRunConfig({ model: msg.model, mode: msg.mode, streams: msg.streams })
        } else if (msg.status === 'stopped') {
          setRunning(false)
          setRunConfig(null)
          setStartedAt(null)
        }
      } else if (msg.type === 'error') {
        setError(`${msg.stream_id || 'stream'}: ${msg.message}`)
      }
    }
    ws.onclose = () => {
      setConnection('disconnected')
      const r = reconnectRef.current
      const delay = Math.min(1000 * 2 ** r.attempts, 8000)
      r.attempts += 1
      r.timer = setTimeout(connect, delay)
    }
    ws.onerror = () => {}
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectRef.current.timer)
      if (wsRef.current) { wsRef.current.onclose = null; wsRef.current.close() }
    }
  }, [connect])

  // ── event detection (action transitions) ─────────────────────────────────
  function detectEvents(msg) {
    const prev = prevActionRef.current[msg.stream_id] || (prevActionRef.current[msg.stream_id] = {})
    const seen = new Set()
    const fresh = []
    for (const t of msg.tracks || []) {
      seen.add(t.id)
      const before = prev[t.id]
      if (before !== t.action) {
        if (before !== undefined || ALERT_ACTIONS.has(t.action)) {
          fresh.push({
            key: `${msg.stream_id}-${t.id}-${msg.frame_number}`,
            ts: Date.now(), stream: msg.stream_id, source: msg.source,
            id: t.id, from: before || null, to: t.action, alert: ALERT_ACTIONS.has(t.action),
          })
        }
        prev[t.id] = t.action
      }
    }
    for (const id of Object.keys(prev)) if (!seen.has(Number(id))) delete prev[id]
    if (fresh.length) setEvents((e) => [...fresh.reverse(), ...e].slice(0, MAX_EVENTS))
  }

  // ── render loop: flush latest frames at most ~30 Hz ──────────────────────
  useEffect(() => {
    let raf
    let last = 0
    const tick = (now) => {
      if (dirtyRef.current && now - last > 32) {
        dirtyRef.current = false
        last = now
        setStreams({ ...latestRef.current })
        setFpsHistory({ ...fpsRef.current })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  // ── 1 Hz occupancy sampler (persons by action, all streams) ──────────────
  useEffect(() => {
    const id = setInterval(() => {
      const frames = Object.values(latestRef.current)
      const stale = Date.now() / 1000 - 3
      const byAction = {}
      let total = 0
      for (const f of frames) {
        if (!f.ts || f.ts < stale) continue
        for (const [a, n] of Object.entries(f.actions || {})) { byAction[a] = (byAction[a] || 0) + n; total += n }
      }
      const arr = occupancyRef.current
      arr.push({ t: Date.now(), total, byAction })
      if (arr.length > OCCUPANCY_SECONDS) arr.shift()
      setOccupancy([...arr])
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // ── controls ─────────────────────────────────────────────────────────────
  const start = useCallback(async (cfg) => {
    setError(null)
    latestRef.current = {}
    fpsRef.current = {}
    prevActionRef.current = {}
    setStreams({})
    setFpsHistory({})
    setEvents([])
    connect()
    try {
      await jsonFetch('/api/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) })
      setRunning(true)
      setStartedAt(Date.now())
      setRunConfig(cfg)
    } catch (e) {
      setError(e.message)
    }
  }, [connect])

  const stop = useCallback(async () => {
    try { await jsonFetch('/api/stop', { method: 'POST' }) } catch (e) { setError(e.message) }
    setRunning(false)
    setRunConfig(null)
    setStartedAt(null)
    setTimeout(() => { latestRef.current = {}; setStreams({}) }, 400)
  }, [])

  const addFolder = useCallback(async (path) => {
    const d = await jsonFetch('/api/folders', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) })
    setVideos(d.videos || [])
  }, [])

  const upload = useCallback(async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const d = await jsonFetch('/api/upload', { method: 'POST', body: fd })
    setVideos(d.videos || [])
    return d.path
  }, [])

  return {
    system, models, videos, streams, fpsHistory, occupancy, events,
    running, runConfig, startedAt, connection, error, backendUp,
    start, stop, addFolder, upload, refreshVideos, clearError: () => setError(null),
  }
}
