import { useCallback, useEffect, useRef, useState } from 'react'
import axios from 'axios'
import './FrameWorkspace.css'

const number = (n) => String(n + 1).padStart(5, '0')
const time = (s = 0) => `${Math.floor(s / 60)}:${(s % 60).toFixed(3).padStart(6, '0')}`
const errorText = (e) => typeof e.response?.data?.detail === 'string' ? e.response.data.detail : e.message
const labels = { manual: 'Locked by you', verified: 'Verified automatically', approximate: 'Approximate · excluded', missing: 'No reliable match' }

function FramePane({ api, job, stream, index, revision, onMove, onLoaded, zoom }) {
  const [frames, setFrames] = useState([])
  const [entry, setEntry] = useState(String(index + 1))
  const [error, setError] = useState('')
  const total = job.media[stream].frame_count
  const prefix = stream === 'low' ? 'A' : 'B'
  const current = frames.find((f) => f.frame_index === index)
  const url = (i, width = 0) => `${api}/jobs/${job.id}/frames/${stream}/${i}?width=${width}&revision=${revision}`
  useEffect(() => {
    const controller = new AbortController()
    axios.get(`${api}/jobs/${job.id}/frame-index/${stream}`, { params: { start: Math.max(0, index - 3), limit: 7 }, signal: controller.signal })
      .then(({ data }) => { setFrames(data.frames); setError('') })
      .catch((e) => { if (!controller.signal.aborted) setError(errorText(e)) })
    return () => controller.abort()
  }, [api, job.id, stream, index])
  useEffect(() => { setEntry(String(index + 1)) }, [index])
  function move(n) { onMove(Math.max(0, Math.min(total - 1, n))) }
  return <section className="exact-pane" aria-label={`${prefix} exact frame viewer`}>
    <header><b>{prefix} · {stream === 'low' ? 'Supplemental' : 'Reference'}</b><span>{job.source_names?.[stream]}</span></header>
    <div className="exact-identity"><strong>Frame {number(index)}</strong><span>{current ? time(current.time_seconds) : 'Loading timestamp…'}</span></div>
    <div className="exact-image-scroll"><img key={`${index}-${revision}`} src={url(index)} style={{ width: `${zoom}%`, maxWidth: 'none' }} alt={`${prefix} exact source frame ${number(index)}`} onLoad={() => onLoaded(stream, index, revision)} onError={() => { setError('Frame could not load. Retry the frame before locking.'); onLoaded(stream, null, revision) }} /></div>
    <div className="exact-jog">
      <button type="button" onClick={() => move(index - 1)} disabled={index === 0} aria-label={`Previous ${prefix} frame`}>← Previous frame</button>
      <form onSubmit={(e) => { e.preventDefault(); const n = Number(entry); if (Number.isInteger(n) && n >= 1 && n <= total) move(n - 1); else setError(`Enter a frame from 1 to ${total}`) }}>
        <label>{prefix} frame <input aria-label={`${prefix} frame number`} type="number" min="1" max={total} value={entry} onChange={(e) => setEntry(e.target.value)} /></label><button type="submit">Go</button>
      </form>
      <button type="button" onClick={() => move(index + 1)} disabled={index === total - 1} aria-label={`Next ${prefix} frame`}>Next frame →</button>
    </div>
    <input aria-label={`${prefix} source position`} type="range" min="0" max={total - 1} value={index} onChange={(e) => move(Number(e.target.value))} />
    <div className="exact-filmstrip">{frames.map((f) => <button key={f.frame_index} type="button" className={f.frame_index === index ? 'selected' : ''} aria-label={`${prefix} frame ${number(f.frame_index)}`} onClick={() => move(f.frame_index)}><img src={url(f.frame_index, 144)} alt="" /><span>{number(f.frame_index)}</span></button>)}</div>
    <small>{current ? `Displayed for ${(current.duration_seconds * 1000).toFixed(2)} ms` : ''} · {job.media[stream].variable_frame_rate ? 'Variable frame timing' : 'Constant frame timing'}</small>
    {error && <p role="alert">{error}</p>}
  </section>
}

export default function FrameWorkspace({ api, job, initialReview, onQueued }) {
  const [review, setReview] = useState(initialReview)
  const [positions, setPositions] = useState({ low: 0, reference: 0 })
  const [loaded, setLoaded] = useState({})
  const [busy, setBusy] = useState(false)
  const [editingLock, setEditingLock] = useState(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('Browsing does not change saved pairs.')
  const [selected, setSelected] = useState(initialReview.ranges[0]?.id || '')
  const [pairs, setPairs] = useState({ total: 0, pairs: [] })
  const [page, setPage] = useState(0)
  const [issuesOnly, setIssuesOnly] = useState(false)
  const [first, setFirst] = useState('')
  const [last, setLast] = useState('')
  const [accepted, setAccepted] = useState(false)
  const [zoom, setZoom] = useState(100)
  const [overlay, setOverlay] = useState(false)
  const [opacity, setOpacity] = useState(50)
  const [framing, setFraming] = useState(review.geometry.framing)
  const [cropText, setCropText] = useState({ low: (review.geometry.crops.low || []).join(', '), reference: (review.geometry.crops.reference || []).join(', ') })
  const [playback, setPlayback] = useState(false)
  const range = review.ranges.find((r) => r.id === selected)
  const requestId = useRef(0)
  const move = useCallback((stream, index) => {
    setPositions((p) => ({ ...p, [stream]: index })); setLoaded((p) => ({ ...p, [stream]: null }))
  }, [])
  const frameLoaded = useCallback((stream, index, revision) => setLoaded((p) => ({ ...p, [stream]: { index, revision } })), [])
  const exactReady = ['low', 'reference'].every((s) => loaded[s]?.index === positions[s] && loaded[s]?.revision === review.revision)
  const alreadyLocked = review.locks.some((p) => p.low_frame === positions.low && p.reference_frame === positions.reference)
  const currentRow = pairs.pairs.findIndex((p) => p.low_frame === positions.low)

  const refresh = async () => {
    try { const { data } = await axios.get(`${api}/jobs/${job.id}/match-review`); setReview(data); setError(''); setNotice('Latest saved review loaded. Your viewer positions were preserved.') }
    catch (e) { setError(errorText(e)) }
  }
  async function edit(payload, message) {
    if (busy) return null
    setBusy(true); setError('')
    try {
      const { data } = await axios.patch(`${api}/jobs/${job.id}/match-review`, { ...payload, revision: review.revision })
      setReview(data); setNotice(message); setAccepted(false)
      return data
    } catch (e) { setError(errorText(e)); return null }
    finally { setBusy(false) }
  }
  function showPair(p) {
    move('low', p.low_frame)
    if (p.reference_frame != null) move('reference', p.reference_frame)
    setNotice(p.reference_frame == null ? 'This A frame has no reliable B match. Browse B to correct it; nothing is locked.' : 'Candidate pair loaded. Saved pairs are unchanged.')
  }
  useEffect(() => {
    const controller = new AbortController(); const id = ++requestId.current
    if (!range?.inspected) { setPairs({ total: 0, pairs: [] }); return () => controller.abort() }
    axios.get(`${api}/jobs/${job.id}/correspondences/${selected}`, { params: { start: page * 100, limit: 100, issues: issuesOnly }, signal: controller.signal })
      .then(({ data }) => { if (id === requestId.current && data.revision === review.revision) setPairs(data) })
      .catch((e) => { if (!controller.signal.aborted) setError(errorText(e)) })
    return () => controller.abort()
  }, [api, job.id, selected, range?.inspected, review.revision, page, issuesOnly])
  useEffect(() => {
    function key(e) {
      if (e.target.closest('input,select,textarea,button,[contenteditable=true]')) return
      const stream = ['ArrowLeft', 'ArrowRight'].includes(e.key) ? 'low' : ['ArrowUp', 'ArrowDown'].includes(e.key) ? 'reference' : null
      if (!stream) return
      e.preventDefault(); const d = ['ArrowLeft', 'ArrowDown'].includes(e.key) ? -1 : 1
      move(stream, Math.max(0, Math.min(review.media[stream].frame_count - 1, positions[stream] + d * (e.shiftKey ? 10 : 1))))
    }
    window.addEventListener('keydown', key); return () => window.removeEventListener('keydown', key)
  }, [positions, review.media, move])
  async function approveProcessing() {
    setBusy(true); setError('')
    try { const { data } = await axios.post(`${api}/jobs/${job.id}/match-review/approve`, { revision: review.revision, mode: review.ranges.length ? 'paired' : 'unpaired' }); onQueued(data) }
    catch (e) { setError(errorText(e)) } finally { setBusy(false) }
  }
  function nextIssue() {
    const next = pairs.pairs.find((p, i) => i > currentRow && !p.eligible)
    if (next) showPair(next)
    else { setIssuesOnly(true); setPage((p) => issuesOnly && (p + 1) * 100 < pairs.total ? p + 1 : 0); setNotice('Showing frames that need attention. Choose a row to inspect it.') }
  }
  const enrichedJob = { ...job, media: review.media, source_names: review.source_names }
  return <div className="frame-workspace">
    <div className="workspace-heading"><div><p className="eyebrow">2 · FRAME MATCHING</p><h2>Two exact frames. One clear commitment.</h2><p>Lock individual pairs first. Approve shared ranges separately.</p></div><button disabled={busy || !review.can_undo} onClick={() => edit({ operation: 'undo' }, 'Last edit undone. Saved pair and range state restored.')}>Undo last edit</button></div>
    {error && <div className="alert error" role="alert">{error} <button onClick={refresh}>Reload saved review</button></div>}
    <div className="frame-view-options"><label>Shared zoom <select value={zoom} onChange={(e) => setZoom(Number(e.target.value))}><option value="100">Fit</option><option value="150">150%</option><option value="200">200%</option><option value="300">300%</option></select></label><label><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> Overlay comparison</label><button onClick={() => setPlayback(!playback)}>{playback ? 'Close playback' : 'Open source playback'}</button></div>
    <div className="exact-viewers">{['low', 'reference'].map((s) => <FramePane key={s} api={api} job={enrichedJob} stream={s} index={positions[s]} revision={review.revision} onMove={(n) => move(s, n)} onLoaded={frameLoaded} zoom={zoom} />)}</div>
    {overlay && <div className="pair-overlay"><div><img src={`${api}/jobs/${job.id}/frames/low/${positions.low}?revision=${review.revision}`} alt="A overlay" /><img style={{ opacity: opacity / 100 }} src={`${api}/jobs/${job.id}/frames/reference/${positions.reference}?revision=${review.revision}`} alt="B overlay" /></div><label>B opacity <input type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} /></label></div>}
    {playback && <div className="source-playback">{['low', 'reference'].map((s) => <figure key={s}><video controls src={`${api}/jobs/${job.id}/navigation/${s}`} /><figcaption>{s === 'low' ? 'A' : 'B'} playback · use exact viewers above to lock frames</figcaption></figure>)}</div>}
    <div className={`pair-commit ${alreadyLocked ? 'locked' : ''}`}>
      <strong>A {number(positions.low)} ↔ B {number(positions.reference)}</strong>
      <button className="primary" disabled={busy || !exactReady || alreadyLocked} onClick={async () => { const result = await edit({ operation: 'lock_pair', low_frame: positions.low, reference_frame: positions.reference, replace_id: editingLock?.id }, `Saved: A ${number(positions.low)} ↔ B ${number(positions.reference)}. Only this pair was locked.`); if (result) setEditingLock(null) }}>{alreadyLocked ? 'This exact pair is locked' : `${editingLock ? 'Replace saved pair with' : 'Lock'} A ${number(positions.low)} ↔ B ${number(positions.reference)}`}</button>
      <span>{!exactReady ? 'Waiting for both exact images…' : 'Locks only these two frames. No range is approved.'}</span>
    </div>
    {editingLock && <p>Editing saved A {number(editingLock.low_frame)} ↔ B {number(editingLock.reference_frame)}. <button onClick={() => setEditingLock(null)}>Cancel pair edit</button></p>}
    <p className="workspace-notice" role="status">{notice}</p>
    <section className="saved-pairs"><h3>Locked by you <span>{review.locks.length}</span></h3>{!review.locks.length && <p>No saved pairs yet. Find the same moment in A and B, then lock the pair above.</p>}
      <div className="saved-pair-list">{review.locks.map((p) => <article key={p.id}><button className="saved-pair-images" onClick={() => { showPair(p); setEditingLock(p) }} aria-label={`Edit pair A ${number(p.low_frame)} B ${number(p.reference_frame)}`}><img src={`${api}/jobs/${job.id}/frames/low/${p.low_frame}?width=144&revision=${review.revision}`} alt={`Saved A ${number(p.low_frame)}`} /><img src={`${api}/jobs/${job.id}/frames/reference/${p.reference_frame}?width=144&revision=${review.revision}`} alt={`Saved B ${number(p.reference_frame)}`} /><b>A {number(p.low_frame)} ↔ B {number(p.reference_frame)}</b><span>Edit pair</span></button><button disabled={busy} onClick={() => edit({ operation: 'remove_pair', low_frame: p.low_frame, reference_frame: p.reference_frame }, 'Pair removed. Affected ranges need another inspection.')}>Remove lock</button></article>)}</div>
      <div className="create-range"><label>First locked pair<select value={first} onChange={(e) => setFirst(e.target.value)}><option value="">Choose first pair</option>{review.locks.map((p) => <option key={p.id} value={p.id}>A {number(p.low_frame)} ↔ B {number(p.reference_frame)}</option>)}</select></label><label>Last locked pair<select value={last} onChange={(e) => setLast(e.target.value)}><option value="">Choose last pair</option>{review.locks.map((p) => <option key={p.id} value={p.id}>A {number(p.low_frame)} ↔ B {number(p.reference_frame)}</option>)}</select></label><button disabled={busy || !first || !last || first === last} onClick={async () => { const data = await edit({ operation: 'create_range', first, last }, 'Range proposed. Inspect its frame pairs before approving.'); if (data) { setSelected(data.ranges.at(-1).id); setPage(0) } }}>Propose range between pairs</button></div>
    </section>
    <section className="range-review"><h3>Shared ranges · approve coverage separately</h3><p>Approved ranges use B in the final edit. Only verified pairs enter training; approximate or missing pairs are excluded.</p>
      <div className="range-layout"><nav aria-label="Shared ranges">{review.ranges.map((r) => <button key={r.id} className={r.id === selected ? 'selected' : ''} onClick={() => { setSelected(r.id); setPage(0); setAccepted(false); showPair({ low_frame: r.low_range[0], reference_frame: r.reference_range[0] }) }}><b>{r.status === 'approved' ? '✓ Approved coverage' : 'Needs review'}</b><span>A {number(r.low_range[0])}–{number(r.low_range[1] - 1)}</span><span>B {number(r.reference_range[0])}–{number(r.reference_range[1] - 1)}</span><small>{r.origin === 'automatic' ? 'Automatic proposal · no manual locks implied' : 'Proposed from your anchors'}</small></button>)}</nav>
        <div>{range ? <>
          <div className="range-actions"><button disabled={busy} onClick={() => edit({ operation: 'inspect_range', range_id: range.id }, 'Frame correspondence inspected. Review exceptions before approving coverage.')}>{range.inspected ? 'Recheck frame pairs' : 'Inspect every frame pair'}</button><button disabled={busy} onClick={() => edit({ operation: 'remove_range', range_id: range.id }, 'Shared range removed. Both source sections will be retained.')}>Keep both sections</button><button onClick={() => showPair({ low_frame: range.low_range[0], reference_frame: range.reference_range[0] })}>Show first pair</button><button onClick={() => showPair({ low_frame: range.low_range[1] - 1, reference_frame: range.reference_range[1] - 1 })}>Show last pair</button></div>
          {range.inspected && <><p className="pair-counts">{range.counts.manual} manual · {range.counts.verified} automatic · {range.counts.approximate} approximate · {range.counts.missing} missing</p><div className="pair-filters"><label><input type="checkbox" checked={issuesOnly} onChange={(e) => { setIssuesOnly(e.target.checked); setPage(0) }} /> Only exceptions</label><button onClick={nextIssue}>Next issue →</button></div><div className="pair-table-scroll"><table><thead><tr><th>A frame</th><th>B frame</th><th>Correspondence</th><th>Training</th></tr></thead><tbody>{pairs.pairs.map((p) => <tr key={p.low_frame} className={positions.low === p.low_frame ? 'current' : ''}><td><button onClick={() => showPair(p)}>{number(p.low_frame)}</button></td><td>{p.reference_frame == null ? '—' : number(p.reference_frame)}</td><td>{labels[p.status]}</td><td>{p.eligible ? 'Eligible after approval' : 'Excluded'}</td></tr>)}</tbody></table></div><div className="pair-pagination"><button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous 100</button><span>{pairs.total ? page * 100 + 1 : 0}–{Math.min((page + 1) * 100, pairs.total)} of {pairs.total}</span><button disabled={(page + 1) * 100 >= pairs.total} onClick={() => setPage(page + 1)}>Next 100</button></div><label className="coverage-ack"><input type="checkbox" checked={accepted} onChange={(e) => setAccepted(e.target.checked)} /> Use B {number(range.reference_range[0])}–{number(range.reference_range[1] - 1)} once in the output instead of this A range. Exclude uncertain frame pairs from training.</label><button className="primary" disabled={busy || !accepted || range.status === 'approved'} onClick={() => edit({ operation: 'approve_range', range_id: range.id, accept_exclusions: true }, 'Range coverage approved. Manual locks remain distinct from automatic matches.')}>{range.status === 'approved' ? 'Range coverage approved' : 'Approve this range'}</button></>}
        </> : <p>No range selected. Lock two pairs to propose one, or retain both sources in full.</p>}</div>
      </div>
    </section>
    <section className="output-storyboard"><h3>Final edit · {time(review.output_duration_seconds)}</h3><p>This is the current export order. Proposed ranges still retain both sources until approved.</p><div>{review.storyboard?.map((c, i) => <button key={i} onClick={() => move(c.source, c.source_range.start_frame)}><b>{c.source === 'low' ? 'A · Restore' : 'B · Reference'}</b><span>{number(c.source_range.start_frame)}–{number(c.source_range.end_frame - 1)}</span><small>{time(c.output_duration_seconds)} · {c.role === 'shared' ? 'Approved overlap' : 'Unique / retained footage'}</small></button>)}</div></section>
    <details className="frame-advanced"><summary>Picture framing and source details</summary><label>Fit to output canvas<select value={framing} onChange={(e) => setFraming(e.target.value)}><option value="fit">Preserve full picture · add bars</option><option value="fill">Crop to fill</option></select></label>{['low', 'reference'].map((s) => <label key={s}>{s === 'low' ? 'A' : 'B'} crop edges: left, top, right, bottom<input value={cropText[s]} placeholder="Blank preserves all source pixels" onChange={(e) => setCropText((p) => ({ ...p, [s]: e.target.value }))} /><small>{review.media[s].width} × {review.media[s].height} · {review.media[s].frame_count} source frames · {time(review.media[s].duration)}{review.media[s].issues.map((v) => ` · ${v}`)}</small></label>)}<button disabled={busy} onClick={() => edit({ operation: 'settings', framing, crops: Object.fromEntries(['low', 'reference'].map((s) => [s, cropText[s].trim() ? cropText[s].split(',').map((n) => Number(n.trim())) : null])) }, 'Framing updated. Inspect affected ranges again; viewers show the applied crop.')}>Apply framing and recheck ranges</button></details>
    <div className="workspace-next"><p>{review.ranges.filter((r) => r.status !== 'approved').length} ranges still need a decision. Locked pairs alone never approve a range.</p><button className="primary" disabled={busy || review.ranges.some((r) => r.status !== 'approved')} onClick={approveProcessing}>Build quality preview →</button></div>
  </div>
}
