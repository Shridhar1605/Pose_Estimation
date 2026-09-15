import { useEffect, useState } from 'react'

function useClock(startedAt) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(id) }, [])
  if (!startedAt) return '00:00:00'
  const s = Math.max(0, Math.floor((now - startedAt) / 1000))
  const hh = String(Math.floor(s / 3600)).padStart(2, '0')
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0')
  const ss = String(s % 60).padStart(2, '0')
  return `${hh}:${mm}:${ss}`
}

function deviceLabel(system) {
  if (!system) return 'detecting hardware…'
  const chip = (system.device_name || '').replace(/^Apple Apple/, 'Apple').replace(/ GPU \(Metal \/ MPS\)$/, '')
  const torch = system.torch_device === 'mps' ? 'MPS' : system.torch_device.startsWith('cuda') ? 'CUDA' : 'CPU'
  const ort = (system.ort_providers_selected?.[0] || 'CPU').replace('ExecutionProvider', '')
  return `${chip} · ${torch} + ${ort}`
}

export default function TopBar({ system, running, runConfig, startedAt, connection }) {
  const clock = useClock(startedAt)
  return (
    <header className="topbar">
      <div className="brand">
        <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
          <circle cx="16" cy="16" r="9" fill="none" stroke="#F5A524" strokeWidth="2.5" />
          <circle cx="16" cy="16" r="3.5" fill="#F5A524" />
          <path d="M4 9V4h5M28 23v5h-5" fill="none" stroke="#F5A524" strokeWidth="1.8" />
        </svg>
        <div>
          <div className="brand-name">Neelaminds <em>Vision</em></div>
        </div>
        <span className="brand-sub">person · pose · action</span>
      </div>

      <div className="topbar-center">
        {running && runConfig ? (
          <>
            <span className="chip live"><span className="dot rec" />LIVE <b className="mono">{clock}</b></span>
            <span className="chip">model <b>{runConfig.model}</b></span>
            <span className="chip">{runConfig.mode === 'multi' ? 'quad 2×2' : 'single feed'}</span>
          </>
        ) : (
          <span className="chip"><span className="dot" />standby</span>
        )}
      </div>

      <div className="topbar-right">
        <span className="chip" title={system ? `${system.platform} · torch ${system.torch} · onnxruntime ${system.onnxruntime}` : ''}>
          {deviceLabel(system)}
        </span>
        <span className="chip" title="WebSocket to backend">
          <span className={`dot ${connection}`} />{connection}
        </span>
      </div>
    </header>
  )
}
