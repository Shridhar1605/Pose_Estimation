import { useEffect, useState } from 'react'
import { usePipeline } from './hooks/usePipeline'
import TopBar from './components/TopBar'
import ControlRail from './components/ControlRail'
import Viewport from './components/Viewport'
import Telemetry from './components/Telemetry'
import OccupancyStrip from './components/OccupancyStrip'
import EventLog from './components/EventLog'

export default function App() {
  const p = usePipeline()
  const [selectedModel, setSelectedModel] = useState('')
  const [mode, setMode] = useState('single')
  const [selected, setSelected] = useState([])

  // sensible defaults once the backend answers
  useEffect(() => { if (!selectedModel && p.models.length) setSelectedModel(p.models[0].name) }, [p.models, selectedModel])
  useEffect(() => {
    if (p.runConfig?.model) setSelectedModel(p.runConfig.model)
    if (p.runConfig?.mode) setMode(p.runConfig.mode)
    if (p.runConfig?.videos) setSelected(p.runConfig.videos)
  }, [p.runConfig])
  useEffect(() => {
    if (selected.length === 0 && p.videos.length && !p.running) setSelected([p.videos[0].path])
  }, [p.videos]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="shell">
      <TopBar system={p.system} running={p.running} runConfig={p.runConfig} startedAt={p.startedAt} connection={p.connection} />
      <ControlRail
        models={p.models} videos={p.videos} running={p.running} error={p.error} clearError={p.clearError} backendUp={p.backendUp}
        selectedModel={selectedModel} setSelectedModel={setSelectedModel}
        mode={mode} setMode={setMode} selected={selected} setSelected={setSelected}
        onStart={p.start} onStop={p.stop} addFolder={p.addFolder} upload={p.upload}
      />
      <Viewport mode={mode} streams={p.streams} selected={selected} running={p.running} />
      <Telemetry streams={p.streams} fpsHistory={p.fpsHistory} running={p.running} />
      <div className="strip">
        <OccupancyStrip occupancy={p.occupancy} />
        <EventLog events={p.events} />
      </div>
    </div>
  )
}
