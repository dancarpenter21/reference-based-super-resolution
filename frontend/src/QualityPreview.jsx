import { useEffect, useRef, useState } from 'react'
import axios from 'axios'

export default function QualityPreview({ api, job, onQueued }) {
  const [data, setData] = useState(null)
  const [method, setMethod] = useState('selected')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [wipe, setWipe] = useState(50)
  const [zoom, setZoom] = useState(100)
  const a = useRef(null), b = useRef(null)
  const origin = api.replace(/\/api\/v1\/?$/, '')
  useEffect(() => {
    const controller = new AbortController()
    axios.get(`${api}/jobs/${job.id}/quality-preview`, { signal: controller.signal }).then(({ data: value }) => {
      setData(value); if (!value.previews.selected) setMethod('lanczos')
    }).catch((e) => { if (!controller.signal.aborted) setError(e.message) })
    return () => controller.abort()
  }, [api, job.id])
  function sync() {
    if (!a.current || !b.current) return
    if (Math.abs(a.current.currentTime - b.current.currentTime) > .08) b.current.currentTime = a.current.currentTime
  }
  async function exportVideo() {
    setBusy(true)
    try { const { data: updated } = await axios.post(`${api}/jobs/${job.id}/render`, { revision: data.revision, method }); onQueued(updated) }
    catch (e) { setError(e.response?.data?.detail || e.message) } finally { setBusy(false) }
  }
  if (!data) return <p>{error || 'Loading quality preview…'}</p>
  return <section className="quality-workspace"><p className="eyebrow">3 · QUALITY PREVIEW</p><h2>Compare before the full export</h2><p>{data.training.reason}. Selected model: <b>{data.training.selected}</b>.</p>
    <div className="quality-controls"><label>Compare Lanczos with<select value={method} onChange={(e) => setMethod(e.target.value)}>{Object.keys(data.previews).map((key) => <option key={key} value={key}>{key === 'selected' ? 'Recommended model' : key === 'adapted' ? 'Adapted candidate · experimental' : key}</option>)}</select></label><label>Zoom<select value={zoom} onChange={(e) => setZoom(Number(e.target.value))}><option value="100">Fit</option><option value="200">200%</option><option value="300">300%</option></select></label></div>
    <div className="quality-scroll"><div className="quality-wipe" style={{ width: `${zoom}%` }}><video ref={a} muted playsInline src={`${origin}${data.previews.lanczos}`} onTimeUpdate={sync} onSeeked={sync} /><video ref={b} muted playsInline style={{ clipPath: `inset(0 0 0 ${wipe}%)` }} src={`${origin}${data.previews[method]}`} onLoadedMetadata={sync} /><span className="before-label">Lanczos</span><span className="after-label">{method}</span></div></div>
    <label>Comparison divider<input aria-label="Comparison divider" type="range" min="0" max="100" value={wipe} onChange={(e) => setWipe(Number(e.target.value))} /></label>
    <div className="quality-controls"><button onClick={() => { sync(); a.current.play().catch(() => {}); b.current.play().catch(() => {}) }}>Play both</button><button onClick={() => { a.current.pause(); b.current.pause() }}>Pause</button><button onClick={() => { a.current.currentTime = 0; b.current.currentTime = 0 }}>Restart</button><label>Position<input aria-label="Preview position" type="range" min="0" max={data.samples.reduce((n, s) => n + s.output_duration_seconds, 0)} step="0.01" defaultValue="0" onChange={(e) => { a.current.currentTime = Number(e.target.value); b.current.currentTime = Number(e.target.value) }} /></label></div>
    <p>Samples: {data.samples.map((s) => s.label).join(' · ')}. Original source timing is preserved.</p>
    <table><thead><tr><th>Real held-out validation</th><th>PSNR</th><th>SSIM</th></tr></thead><tbody>{Object.entries(data.training.real_validation || {}).map(([name, value]) => value && <tr key={name}><td>{name}</td><td>{value.psnr.toFixed(2)} dB</td><td>{value.ssim.toFixed(4)}</td></tr>)}</tbody></table>
    <p>{data.output_geometry.width} × {data.output_geometry.height} · Full edit {(data.output_duration_seconds / 60).toFixed(2)} minutes · Original frame timing</p>
    {error && <p role="alert">{error}</p>}
    {job.state === 'awaiting_quality_preview' && <button className="primary" disabled={busy} onClick={exportVideo}>{busy ? 'Queuing export…' : `Export complete edit with ${method}`}</button>}
  </section>
}
