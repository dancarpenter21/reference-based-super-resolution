import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const terminal = new Set(['completed', 'failed', 'cancelled'])

function FileField({ label, description, file, onChange }) {
  return (
    <label className="file-field">
      <span className="file-title">{label}</span>
      <span className="file-help">{description}</span>
      <input type="file" accept="video/mp4,video/quicktime,.m4v" onChange={(event) => onChange(event.target.files?.[0] || null)} />
      <span className={file ? 'file-name selected' : 'file-name'}>{file ? file.name : 'Choose MP4 or MOV'}</span>
    </label>
  )
}

function App() {
  const [low, setLow] = useState(null)
  const [reference, setReference] = useState(null)
  const [preset, setPreset] = useState('balanced')
  const [job, setJob] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState('')
  const jobId = job?.id
  const jobState = job?.state

  useEffect(() => {
    if (!jobId || terminal.has(jobState)) return undefined
    const timer = window.setInterval(async () => {
      try {
        const response = await axios.get(`${API}/jobs/${jobId}`)
        setJob(response.data)
      } catch (requestError) {
        setError(requestError.response?.data?.detail || requestError.message)
      }
    }, 1500)
    return () => window.clearInterval(timer)
  }, [jobId, jobState])

  const busy = Boolean(job && !terminal.has(job.state))
  const progress = Math.round((job?.progress || 0) * 100)
  const eta = job?.eta_seconds ? `${Math.max(1, Math.round(job.eta_seconds / 60))} min remaining` : null

  async function submit(event) {
    event.preventDefault()
    if (!low || !reference) {
      setError('Select both the complete low-resolution video and the high-resolution reference.')
      return
    }
    setError('')
    setUploadProgress(0)
    const data = new FormData()
    data.append('low_video', low)
    data.append('reference_video', reference)
    data.append('preset', preset)
    try {
      const response = await axios.post(`${API}/jobs`, data, {
        onUploadProgress: ({ loaded, total }) => setUploadProgress(total ? Math.round(loaded * 100 / total) : 0),
      })
      setJob(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function cancel() {
    const response = await axios.post(`${API}/jobs/${job.id}/cancel`)
    setJob(response.data)
  }

  async function remove() {
    await axios.delete(`${API}/jobs/${job.id}`)
    setJob(null)
    setUploadProgress(0)
  }

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">LOCAL GPU RESTORATION</p>
        <h1>Recover the complete cut.<br /><span>Keep the better detail.</span></h1>
        <p className="lede">Adapt a faithful upscaler from an incomplete high-resolution reference, then restore every frame of the complete source.</p>
      </header>

      <section className="panel">
        <form onSubmit={submit}>
          <div className="files">
            <FileField label="01 · Complete source" description="The full low-resolution video. Its timing, frames, and audio are preserved." file={low} onChange={setLow} />
            <FileField label="02 · Detail reference" description="The incomplete high-resolution footage used to adapt the restoration model." file={reference} onChange={setReference} />
          </div>
          <div className="controls">
            <label>
              <span>Training preset</span>
              <select value={preset} onChange={(event) => setPreset(event.target.value)} disabled={busy}>
                <option value="quick">Quick · up to 15 min</option>
                <option value="balanced">Balanced · up to 1 hour</option>
                <option value="quality">Quality · up to 4 hours</option>
              </select>
            </label>
            <button className="primary" disabled={busy} type="submit">{busy ? 'GPU job running' : 'Analyze and upscale'}</button>
          </div>
        </form>
        {uploadProgress > 0 && !job && <p className="upload">Uploading · {uploadProgress}%</p>}
        {error && <p className="alert error" role="alert">{error}</p>}
      </section>

      {job && (
        <section className="panel job" aria-live="polite">
          <div className="job-head">
            <div><p className="eyebrow">JOB {job.id.slice(0, 8)}</p><h2>{job.stage.replace('_', ' ')}</h2></div>
            <strong>{progress}%</strong>
          </div>
          <div className="progress"><span style={{ width: `${progress}%` }} /></div>
          <p className="message">{job.message}{eta ? ` · ${eta}` : ''}</p>
          {job.alignment_mode && <p className="mode">Alignment mode · <b>{job.alignment_mode}</b></p>}
          {job.warning && <p className="alert warning">{job.warning}</p>}
          {job.error && <p className="alert error">{job.error}</p>}
          {job.metrics?.psnr && <div className="metrics"><span>Validation PSNR</span><b>{job.metrics.psnr.toFixed(2)} dB</b></div>}
          {job.state === 'completed' && (
            <div className="result">
              <video controls src={`${API.replace('/api/v1', '')}${job.result_url}`} />
              <div className="actions">
                <a className="primary link" href={`${API.replace('/api/v1', '')}${job.result_url}`} download>Download restored video</a>
                <a href={`${API.replace('/api/v1', '')}${job.report_url}`} download>Download report</a>
              </div>
            </div>
          )}
          <div className="job-actions">
            {!terminal.has(job.state) && <button onClick={cancel}>Cancel job</button>}
            {terminal.has(job.state) && <button className="danger" onClick={remove}>Delete job and files</button>}
          </div>
        </section>
      )}
    </main>
  )
}

export default App
