import { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const terminal = new Set(['completed', 'failed', 'cancelled'])
const selectedJobKey = 'refsr-selected-job'

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

function SystemStatus({ status, online, onRecheck }) {
  const worker = status?.worker
  const gpu = status?.gpu
  const queue = status?.queue
  const workerText = worker?.state === 'busy'
    ? `Processing job ${worker.current_job_id.slice(0, 8)}`
    : worker?.state === 'idle' ? 'Idle and ready' : 'Stopped'
  const gpuText = gpu?.state === 'available'
    ? gpu.name
    : gpu?.state === 'unavailable' ? 'Unavailable' : 'Checking availability'

  return (
    <section className="panel system-panel" aria-label="System status">
      <div className="system-heading">
        <div><p className="eyebrow">SYSTEM STATUS</p><h2>Processing capacity</h2></div>
        <button onClick={onRecheck} disabled={online !== true || gpu?.state === 'checking'} type="button">Recheck GPU</button>
      </div>
      <div className="status-grid">
        <div><span>Backend</span><b className={online === false ? 'status-bad' : online === true ? 'status-good' : ''}>{online === false ? 'Unreachable' : online === true ? 'Online' : 'Connecting'}</b></div>
        <div><span>Worker</span><b className={worker?.state === 'stopped' ? 'status-bad' : ''}>{online === false ? 'Unknown' : workerText}</b></div>
        <div><span>GPU</span><b className={gpu?.state === 'available' ? 'status-good' : gpu?.state === 'unavailable' ? 'status-bad' : ''}>{online === false ? 'Unknown' : gpuText}</b></div>
        <div><span>Outstanding</span><b>{online === false ? 'Unknown' : `${queue?.outstanding ?? 0} (${queue?.queued ?? 0} queued)`}</b></div>
      </div>
      {online === false && <p className="alert error">The backend is unreachable. Existing jobs remain visible, but no worker can claim queued work until the backend is running.</p>}
      {worker?.fatal_error && <p className="alert error">Worker stopped: {worker.fatal_error}</p>}
      {gpu?.state === 'available' && <p className="status-detail">{gpu.device_count} device{gpu.device_count === 1 ? '' : 's'} · {gpu.device} · HIP {gpu.hip_version} · PyTorch {gpu.torch_version}</p>}
      {gpu?.state === 'unavailable' && <p className="alert error">GPU check failed: {gpu.error}</p>}
    </section>
  )
}

function JobList({ jobs, selectedId, onSelect, onCancelAll }) {
  const activeCount = jobs.filter((job) => !terminal.has(job.state)).length

  return (
    <section className="panel jobs-panel" aria-label="Jobs">
      <div className="jobs-heading">
        <div><p className="eyebrow">DURABLE QUEUE</p><h2>Jobs</h2></div>
        {activeCount > 0 && <button className="danger" onClick={onCancelAll}>Cancel all active jobs</button>}
      </div>
      {jobs.length === 0 ? (
        <p className="empty-jobs">No jobs yet. Submitted work will remain available here after a restart.</p>
      ) : (
        <div className="job-list">
          {jobs.map((job) => {
            const progress = Math.round((job.progress || 0) * 100)
            return (
              <button
                className={`job-row${job.id === selectedId ? ' selected' : ''}`}
                key={job.id}
                onClick={() => onSelect(job.id)}
                type="button"
              >
                <span><b>JOB {job.id.slice(0, 8)}</b><small>{job.preset}</small></span>
                <span><b className={`state state-${job.state}`}>{job.stage.replace('_', ' ')}</b><small>{new Date(job.created_at).toLocaleString()}</small></span>
                <strong>{progress}%</strong>
              </button>
            )
          })}
        </div>
      )}
    </section>
  )
}

function App() {
  const [low, setLow] = useState(null)
  const [reference, setReference] = useState(null)
  const [preset, setPreset] = useState('balanced')
  const [jobs, setJobs] = useState([])
  const [jobsLoaded, setJobsLoaded] = useState(false)
  const [selectedId, setSelectedId] = useState(() => window.localStorage.getItem(selectedJobKey))
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState('')
  const [jobsError, setJobsError] = useState('')
  const [system, setSystem] = useState(null)
  const [backendOnline, setBackendOnline] = useState(null)

  const refreshJobs = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/jobs`)
      setJobs(response.data.jobs)
      setJobsLoaded(true)
      setJobsError('')
    } catch (requestError) {
      setJobsError(requestError.response?.data?.detail || requestError.message)
    }
  }, [])

  const refreshSystem = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/system/status`)
      setSystem(response.data)
      setBackendOnline(true)
    } catch {
      setBackendOnline(false)
    }
  }, [])

  useEffect(() => {
    const initial = window.setTimeout(refreshJobs, 0)
    const initialSystem = window.setTimeout(refreshSystem, 0)
    const timer = window.setInterval(refreshJobs, 1500)
    const systemTimer = window.setInterval(refreshSystem, 1500)
    return () => {
      window.clearTimeout(initial)
      window.clearTimeout(initialSystem)
      window.clearInterval(timer)
      window.clearInterval(systemTimer)
    }
  }, [refreshJobs, refreshSystem])

  const effectiveSelectedId = useMemo(() => {
    if (selectedId && jobs.some((job) => job.id === selectedId)) return selectedId
    return (jobs.find((job) => !terminal.has(job.state)) || jobs[0])?.id || null
  }, [jobs, selectedId])

  useEffect(() => {
    if (!jobsLoaded) return
    if (effectiveSelectedId) window.localStorage.setItem(selectedJobKey, effectiveSelectedId)
    else window.localStorage.removeItem(selectedJobKey)
  }, [effectiveSelectedId, jobsLoaded])

  const job = useMemo(() => jobs.find((item) => item.id === effectiveSelectedId) || null, [effectiveSelectedId, jobs])
  const busy = jobs.some((item) => !terminal.has(item.state))
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
      setJobs((current) => [response.data, ...current.filter((item) => item.id !== response.data.id)])
      setSelectedId(response.data.id)
      setUploadProgress(0)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function cancel() {
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/cancel`)
      setJobs((current) => current.map((item) => item.id === response.data.id ? response.data : item))
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function cancelAll() {
    if (!window.confirm('Cancel every queued or running job?')) return
    try {
      const response = await axios.post(`${API}/jobs/cancel-all`)
      setJobs(response.data.jobs)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function recheckGpu() {
    try {
      const response = await axios.post(`${API}/system/gpu/recheck`)
      setSystem((current) => current ? { ...current, gpu: response.data } : current)
      setBackendOnline(true)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
      setBackendOnline(false)
    }
  }

  async function remove() {
    try {
      await axios.delete(`${API}/jobs/${job.id}`)
      setJobs((current) => current.filter((item) => item.id !== job.id))
      setUploadProgress(0)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">LOCAL GPU RESTORATION</p>
        <h1>Recover the complete cut.<br /><span>Keep the better detail.</span></h1>
        <p className="lede">Adapt a faithful upscaler from an incomplete high-resolution reference, then restore every frame of the complete source.</p>
      </header>

      <SystemStatus status={system} online={backendOnline} onRecheck={recheckGpu} />

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
            <button className="primary" disabled={busy} type="submit">{busy ? 'Work already queued or processing' : 'Analyze and upscale'}</button>
          </div>
        </form>
        {uploadProgress > 0 && <p className="upload">Uploading · {uploadProgress}%</p>}
        {jobsError && <p className="alert error" role="alert">Job list refresh failed: {jobsError}</p>}
        {error && <p className="alert error" role="alert">{error}</p>}
      </section>

      <JobList jobs={jobs} selectedId={effectiveSelectedId} onSelect={setSelectedId} onCancelAll={cancelAll} />

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
