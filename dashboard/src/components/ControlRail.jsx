import { useMemo, useRef, useState } from 'react'

function fmtMeta(v) {
  const parts = []
  if (v.width && v.height) parts.push(`${v.width}×${v.height}`)
  if (v.fps) parts.push(`${v.fps}fps`)
  if (v.duration_s) parts.push(`${Math.round(v.duration_s)}s`)
  return parts.join(' · ')
}

export default function ControlRail({
  models, videos, running, error, clearError, backendUp,
  selectedModel, setSelectedModel, mode, setMode, selected, setSelected,
  onStart, onStop, addFolder, upload,
}) {
  const [url, setUrl] = useState('')
  const [folder, setFolder] = useState('')
  const [busy, setBusy] = useState(false)
  const fileRef = useRef(null)
  const max = mode === 'single' ? 1 : 4

  const grouped = useMemo(() => {
    const g = {}
    for (const v of videos) (g[v.folder || ''] ||= []).push(v)
    return g
  }, [videos])

  const extras = selected.filter((p) => !videos.some((v) => v.path === p))

  const toggle = (path) => {
    if (running) return
    if (mode === 'single') return setSelected([path])
    setSelected((s) => (s.includes(path) ? s.filter((p) => p !== path) : s.length >= max ? s : [...s, path]))
  }
  const changeMode = (m) => { setMode(m); setSelected((s) => s.slice(0, m === 'single' ? 1 : 4)) }
  const addSource = (src) => {
    if (!src) return
    if (mode === 'single') setSelected([src])
    else setSelected((s) => (s.includes(src) || s.length >= max ? s : [...s, src]))
  }
  const onUpload = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setBusy(true)
    try { const p = await upload(f); addSource(p) } catch (err) { alert(err.message) } finally { setBusy(false); e.target.value = '' }
  }
  const canStart = backendUp && selectedModel && selected.length > 0 && !running

  return (
    <aside className="rail">
      <div className="rail-scroll">
        <section>
          <div className="section-head"><span className="eyebrow">Detector</span><span className="meta">{models.length} available</span></div>
          <div className="model-list">
            {models.map((m) => (
              <button key={m.name} className={`model-card ${selectedModel === m.name ? 'selected' : ''}`}
                onClick={() => setSelectedModel(m.name)} disabled={running} aria-pressed={selectedModel === m.name}>
                <div className="row"><span className="name">{m.name}</span><span className={`tag ${m.type}`}>{m.type === 'yolo' ? 'YOLO26' : 'PeopleNet'}</span></div>
                <div className="row"><span className="desc">{m.backend}</span><span className="desc mono">{m.size_mb} MB</span></div>
              </button>
            ))}
            {models.length === 0 && <div className="empty-note">No models found. Run <span className="mono">python download_models.py</span>.</div>}
          </div>
        </section>

        <section>
          <div className="section-head"><span className="eyebrow">Layout</span></div>
          <div className="segmented" role="radiogroup">
            <button className={mode === 'single' ? 'active' : ''} onClick={() => changeMode('single')} disabled={running}>Single feed</button>
            <button className={mode === 'multi' ? 'active' : ''} onClick={() => changeMode('multi')} disabled={running}>Quad 2×2</button>
          </div>
        </section>

        <section>
          <div className="section-head"><span className="eyebrow">Sources</span><span className="meta mono">{selected.length}/{max} selected</span></div>
          <div className="source-list">
            {Object.entries(grouped).map(([folder, list]) => (
              <div key={folder}>
                {Object.keys(grouped).length > 1 && <div className="source-folder">{folder || '/'}</div>}
                {list.map((v) => {
                  const on = selected.includes(v.path)
                  const off = running || (!on && selected.length >= max && mode === 'multi')
                  return (
                    <label key={v.path} className={`source ${on ? 'selected' : ''} ${off && !on ? 'disabled' : ''}`}>
                      <input type="checkbox" checked={on} disabled={off && !on} onChange={() => toggle(v.path)} />
                      <span><div className="sname" title={v.path}>{v.name}</div><div className="smeta">{fmtMeta(v)}</div></span>
                      <span className="slot">{on ? `CAM ${String(selected.indexOf(v.path) + 1).padStart(2, '0')}` : ''}</span>
                    </label>
                  )
                })}
              </div>
            ))}
            {extras.map((p) => (
              <label key={p} className="source selected">
                <input type="checkbox" checked onChange={() => toggle(p)} disabled={running} />
                <span><div className="sname" title={p}>{p}</div><div className="smeta">{p.startsWith('webcam') || /^\d+$/.test(p) ? 'camera' : 'network stream'}</div></span>
                <span className="slot">CAM {String(selected.indexOf(p) + 1).padStart(2, '0')}</span>
              </label>
            ))}
            {videos.length === 0 && backendUp && <div className="empty-note">No videos found. Drop files into <span className="mono">Video_samples/</span>, upload one, or add a stream URL.</div>}
          </div>

          <div className="add-row">
            <input className="input mono" placeholder="rtsp://… or webcam:0" value={url} onChange={(e) => setUrl(e.target.value)} disabled={running}
              onKeyDown={(e) => { if (e.key === 'Enter') { addSource(url.trim()); setUrl('') } }} />
            <button className="btn" disabled={running || !url.trim()} onClick={() => { addSource(url.trim()); setUrl('') }}>Add</button>
          </div>
          <div className="add-row">
            <button className="btn ghost" style={{ flex: 1 }} disabled={running || busy} onClick={() => fileRef.current?.click()}>{busy ? 'Uploading…' : 'Upload video'}</button>
            <button className="btn ghost" style={{ flex: 1 }} disabled={running} onClick={() => addSource('webcam:0')}>Use webcam</button>
            <input ref={fileRef} type="file" accept="video/*,.mkv,.avi,.mov,.mp4" hidden onChange={onUpload} />
          </div>
          <div className="add-row">
            <input className="input mono" placeholder="/path/to/folder of videos" value={folder} onChange={(e) => setFolder(e.target.value)} disabled={running} />
            <button className="btn" disabled={running || !folder.trim()} onClick={async () => { try { await addFolder(folder.trim()); setFolder('') } catch (e) { alert(e.message) } }}>Scan</button>
          </div>
        </section>
      </div>

      <div className="rail-foot">
        {error && <div className="error" role="alert"><span>⚠</span><span>{error}</span><button onClick={clearError} aria-label="Dismiss">✕</button></div>}
        {!running ? (
          <button className="btn-run" disabled={!canStart} onClick={() => onStart({ model: selectedModel, mode, videos: selected })}>▶ Start pipeline</button>
        ) : (
          <button className="btn-run stop" onClick={onStop}>■ Stop pipeline</button>
        )}
      </div>
    </aside>
  )
}
