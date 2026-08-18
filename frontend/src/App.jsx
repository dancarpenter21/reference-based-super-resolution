import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const terminal = new Set(['completed', 'failed', 'cancelled'])
const selectedJobKey = 'refsr-selected-job'
const ORIGIN = API.replace('/api/v1', '')

function matchingLabel(mode) {
  return mode === 'reference_only' ? 'reference only' : 'guided matching'
}

function formatTime(seconds = 0) {
  const whole = Math.max(0, seconds)
  const hours = Math.floor(whole / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  const secs = (whole % 60).toFixed(3).padStart(6, '0')
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secs}`
}

function MatchReview({ job, onQueued }) {
  const [review, setReview] = useState(null)
  const [selected, setSelected] = useState(0)
  const [boundary, setBoundary] = useState('start')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [overlay, setOverlay] = useState(0)
  const lowVideo = useRef(null)
  const referenceVideo = useRef(null)

  const load = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/jobs/${job.id}/match-review`)
      setReview(response.data)
      setError('')
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }, [job.id])

  useEffect(() => { load() }, [load])
  const segment = review?.segments?.[selected] || null

  async function persist(segments) {
    setSaving(true)
    try {
      const response = await axios.put(`${API}/jobs/${job.id}/match-review`, {
        revision: review.revision, segments,
      })
      setReview((current) => ({ ...current, ...response.data }))
      setError('')
      return response.data
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
      return null
    } finally {
      setSaving(false)
    }
  }

  function changedSegment(mutator) {
    const segments = review.segments.map((item, index) => index === selected ? mutator(item) : item)
    persist(segments)
  }

  async function resolveSelected(status) {
    const segments = review.segments.map((item, index) => index === selected ? { ...item, status } : item)
    const saved = await persist(segments)
    if (!saved) return
    const next = saved.segments.findIndex((item, index) => index > selected && item.status === 'proposed')
    if (next >= 0) setSelected(next)
    else {
      const first = saved.segments.findIndex((item) => item.status === 'proposed')
      if (first >= 0) setSelected(first)
    }
  }

  function step(stream, amount) {
    if (!segment || saving) return
    const key = `${stream}_${boundary}`
    const info = review.media[stream]
    changedSegment((item) => {
      const next = Math.max(0, Math.min(info.frame_count - 1, item[key].frame_index + amount))
      return { ...item, [key]: { frame_index: next, pts: next, time_seconds: next / info.fps }, status: 'proposed', origin: 'manual' }
    })
  }

  useEffect(() => {
    function keydown(event) {
      if (!segment || event.target.matches('input, button, select')) return
      const amount = event.shiftKey ? 10 : 1
      if (event.key === 'ArrowLeft') { event.preventDefault(); step('low', -amount) }
      if (event.key === 'ArrowRight') { event.preventDefault(); step('low', amount) }
      if (event.key === 'ArrowDown') { event.preventDefault(); step('reference', -amount) }
      if (event.key === 'ArrowUp') { event.preventDefault(); step('reference', amount) }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  })

  async function addFromPlayheads() {
    const lowFrame = Math.round((lowVideo.current?.currentTime || 0) * review.media.low.fps)
    const referenceFrame = Math.round((referenceVideo.current?.currentTime || 0) * review.media.reference.fps)
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/expand`, {
        low_frame_index: lowFrame, reference_frame_index: referenceFrame,
      })
      const saved = await persist([...review.segments, response.data])
      if (saved) setSelected(saved.segments.length - 1)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  function splitAtPlayheads() {
    if (!segment) return
    const lowIndex = Math.round((lowVideo.current?.currentTime || segment.low_start.time_seconds) * review.media.low.fps)
    const refIndex = Math.round((referenceVideo.current?.currentTime || segment.reference_start.time_seconds) * review.media.reference.fps)
    if (lowIndex <= segment.low_start.frame_index || lowIndex >= segment.low_end.frame_index || refIndex <= segment.reference_start.frame_index || refIndex >= segment.reference_end.frame_index) {
      setError('Move both playheads inside the selected segment before splitting.')
      return
    }
    const lowRef = { frame_index: lowIndex, pts: lowIndex, time_seconds: lowIndex / review.media.low.fps }
    const highRef = { frame_index: refIndex, pts: refIndex, time_seconds: refIndex / review.media.reference.fps }
    const left = { ...segment, id: `${segment.id}-a`, low_end: lowRef, reference_end: highRef, status: 'proposed', origin: 'manual' }
    const right = { ...segment, id: `${segment.id}-b`, low_start: lowRef, reference_start: highRef, status: 'proposed', origin: 'manual' }
    persist(review.segments.flatMap((item, index) => index === selected ? [left, right] : [item]))
  }

  function mergeNext() {
    if (!segment || selected >= review.segments.length - 1) return
    const next = review.segments[selected + 1]
    const merged = { ...segment, id: `${segment.id}-merged`, low_end: next.low_end, reference_end: next.reference_end, status: 'proposed', origin: 'manual' }
    persist(review.segments.filter((_, index) => index !== selected + 1).map((item, index) => index === selected ? merged : item))
  }

  async function snap(targetStream) {
    if (!segment || saving) return
    const fixedStream = targetStream === 'low' ? 'reference' : 'low'
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/snap`, {
        fixed_stream: fixedStream,
        fixed_frame_index: segment[`${fixedStream}_${boundary}`].frame_index,
        target_frame_index: segment[`${targetStream}_${boundary}`].frame_index,
      })
      const segments = review.segments.map((item, index) => index === selected ? {
        ...item, [`${targetStream}_${boundary}`]: response.data.frame, status: 'proposed', origin: 'manual',
      } : item)
      await persist(segments)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  async function approve(mode) {
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/approve`, { revision: review.revision, mode })
      onQueued(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  if (!review) return <p className="message">Loading frame-match workspace…</p>
  const unresolved = review.segments.filter((item) => item.status === 'proposed').length
  const confirmed = review.segments.filter((item) => item.status === 'confirmed').length
  const lowDuration = review.media.low.duration || 1
  const refDuration = review.media.reference.duration || 1
  const imageUrl = (stream, ref) => `${API}/jobs/${job.id}/frames/${stream}/${ref.frame_index}`

  return (
    <div className="match-review">
      <div className="review-summary">
        <div><b>{confirmed}</b><span>confirmed</span></div>
        <div><b>{unresolved}</b><span>needs review</span></div>
        <div><b>{formatTime(review.summary?.matched_seconds || 0)}</b><span>matched</span></div>
      </div>
      <div className="timeline" aria-label="Complete source match timeline">
        <span>Complete source</span><div>{review.segments.map((item) => <i key={item.id} className={`range ${item.status}`} style={{ left: `${item.low_start.time_seconds / lowDuration * 100}%`, width: `${Math.max(.4, (item.low_end.time_seconds - item.low_start.time_seconds) / lowDuration * 100)}%` }} />)}</div>
        <span>Reference</span><div>{review.segments.map((item) => <i key={item.id} className={`range ${item.status}`} style={{ left: `${item.reference_start.time_seconds / refDuration * 100}%`, width: `${Math.max(.4, (item.reference_end.time_seconds - item.reference_start.time_seconds) / refDuration * 100)}%` }} />)}</div>
      </div>
      <div className="navigation-videos">
        <label>Complete source<video ref={lowVideo} controls src={`${ORIGIN}${review.proxy_urls.low}`} /></label>
        <label>Reference<video ref={referenceVideo} controls src={`${ORIGIN}${review.proxy_urls.reference}`} /></label>
      </div>
      <div className="segment-toolbar">
        <select aria-label="Review segment" value={selected} onChange={(event) => setSelected(Number(event.target.value))}>
          {review.segments.map((item, index) => <option key={item.id} value={index}>Segment {index + 1} · {item.status}</option>)}
        </select>
        <button onClick={addFromPlayheads} disabled={saving}>Add from playheads</button>
        <button onClick={splitAtPlayheads} disabled={!segment || saving}>Split at playheads</button>
        <button onClick={mergeNext} disabled={!segment || selected >= review.segments.length - 1 || saving}>Merge next</button>
      </div>
      {segment && <>
        <div className="boundary-tabs"><button className={boundary === 'start' ? 'selected' : ''} onClick={() => setBoundary('start')}>Start frames</button><button className={boundary === 'end' ? 'selected' : ''} onClick={() => setBoundary('end')}>End frames</button></div>
        <div className={`frame-compare${overlay ? ' overlay' : ''}`} style={{ '--overlay': overlay / 100 }}>
          {['low', 'reference'].map((stream) => {
            const ref = segment[`${stream}_${boundary}`]
            return <div className={`frame-pane pane-${stream}`} key={stream}>
              <span>{stream === 'low' ? 'Complete source' : 'Reference'} · frame {ref.frame_index} · {formatTime(ref.time_seconds)}</span>
              <img src={imageUrl(stream, ref)} alt={`${stream} ${boundary} frame`} />
              <div className="step-controls">{[-10, -1, 1, 10].map((amount) => <button key={amount} disabled={saving} onClick={() => step(stream, amount)}>{amount > 0 ? '+' : ''}{amount}</button>)}</div>
              <button className="snap" disabled={saving} onClick={() => snap(stream)}>Snap to other frame</button>
            </div>
          })}
        </div>
        <label className="overlay-control">Overlay comparison <input type="range" min="0" max="100" value={overlay} onChange={(event) => setOverlay(Number(event.target.value))} /></label>
        <div className="review-actions">
          <button onClick={() => resolveSelected('rejected')} disabled={saving}>Reject segment</button>
          <button className="primary" onClick={() => resolveSelected('confirmed')} disabled={saving}>Confirm boundaries</button>
        </div>
      </>}
      {error && <p className="alert error" role="alert">{error}</p>}
      <div className="approval-actions">
        <button onClick={() => approve('unpaired')} disabled={saving || unresolved > 0}>Continue without paired matches</button>
        <button className="primary" onClick={() => approve('paired')} disabled={saving || unresolved > 0 || confirmed === 0}>Approve matches and start processing</button>
      </div>
    </div>
  )
}

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
        <div><span>Outstanding</span><b>{online === false ? 'Unknown' : `${queue?.outstanding ?? 0} (${queue?.queued ?? 0} queued · ${queue?.needs_review ?? 0} review)`}</b></div>
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
                <span><b>JOB {job.id.slice(0, 8)}</b><small>{job.preset} · {matchingLabel(job.matching_mode)}</small></span>
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
  const [matchingMode, setMatchingMode] = useState('guided')
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
  const busy = jobs.some((item) => !terminal.has(item.state) && item.state !== 'awaiting_match_review')
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
    data.append('matching_mode', matchingMode)
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

  function reviewQueued(updated) {
    setJobs((current) => current.map((item) => item.id === updated.id ? updated : item))
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
          <fieldset className="mode-options">
            <legend>How should the reference be used?</legend>
            <label className={matchingMode === 'guided' ? 'selected' : ''}>
              <input type="radio" name="matching-mode" value="guided" checked={matchingMode === 'guided'} onChange={(event) => setMatchingMode(event.target.value)} disabled={busy} />
              <span><b>Find and match shared frames</b><small>Recommended · Review exact matching sections before training for the strongest supervision.</small></span>
            </label>
            <label className={matchingMode === 'reference_only' ? 'selected' : ''}>
              <input type="radio" name="matching-mode" value="reference_only" checked={matchingMode === 'reference_only'} onChange={(event) => setMatchingMode(event.target.value)} disabled={busy} />
              <span><b>Skip matching · reference only</b><small>Start sooner using synthetic low-quality pairs when the videos do not share trustworthy frames.</small></span>
            </label>
          </fieldset>
          <div className="controls">
            <label>
              <span>Training preset</span>
              <select value={preset} onChange={(event) => setPreset(event.target.value)} disabled={busy}>
                <option value="quick">Quick · up to 15 min</option>
                <option value="balanced">Balanced · up to 1 hour</option>
                <option value="quality">Quality · up to 4 hours</option>
              </select>
            </label>
            <button className="primary" disabled={busy} type="submit">{busy ? 'Work already queued or processing' : matchingMode === 'guided' ? 'Analyze frames' : 'Start reference-only processing'}</button>
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
          <p className="mode">Workflow · <b>{matchingLabel(job.matching_mode)}</b></p>
          {job.alignment_mode && <p className="mode">Alignment mode · <b>{job.alignment_mode}</b></p>}
          {job.warning && <p className="alert warning">{job.warning}</p>}
          {job.error && <p className="alert error">{job.error}</p>}
          {job.metrics?.psnr && <div className="metrics"><span>Validation PSNR</span><b>{job.metrics.psnr.toFixed(2)} dB</b></div>}
          {job.state === 'awaiting_match_review' && <MatchReview job={job} onQueued={reviewQueued} />}
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
