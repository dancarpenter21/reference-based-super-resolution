import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

vi.mock('axios')

const activeJob = {
  id: 'active-job-1234', state: 'training', stage: 'training', progress: 0.4,
  message: 'step 40', preset: 'balanced', metrics: null, warning: null, error: null,
  created_at: '2026-08-13T12:00:00+00:00', updated_at: '2026-08-13T12:01:00+00:00',
  eta_seconds: 120,
}

const completedJob = {
  id: 'done-job-5678', state: 'completed', stage: 'completed', progress: 1,
  message: 'Upscaled video is ready', preset: 'quick', metrics: null, warning: null, error: null,
  created_at: '2026-08-12T12:00:00+00:00', updated_at: '2026-08-12T13:00:00+00:00',
  result_url: '/api/v1/jobs/done-job-5678/result', report_url: '/api/v1/jobs/done-job-5678/report',
}

const systemStatus = {
  backend: { state: 'online' },
  worker: { state: 'idle', current_job_id: null, started_at: '2026-08-13T12:00:00Z', fatal_error: null },
  gpu: {
    state: 'available', device: 'cuda:0', device_count: 1, name: 'AMD Radeon RX 9070 XT',
    hip_version: '7.2', torch_version: '2.9.1', error: null, last_checked_at: '2026-08-13T12:00:01Z',
  },
  queue: { queued: 0, processing: 0, outstanding: 0 },
}

function mockGets(jobs = [], status = systemStatus) {
  axios.get.mockImplementation((url) => Promise.resolve({ data: url.endsWith('/system/status') ? status : { jobs } }))
}

describe('App', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
    mockGets()
  })

  it('renders the two-video workflow and an empty durable queue', async () => {
    render(<App />)
    expect(screen.getByText('01 · Complete source')).toBeInTheDocument()
    expect(screen.getByText('02 · Detail reference')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /find and match shared frames/i })).toBeChecked()
    expect(screen.getByRole('button', { name: /analyze frames/i })).toBeInTheDocument()
    expect(await screen.findByText(/No jobs yet/)).toBeInTheDocument()
  })

  it('restores jobs, selects the active job, and disables submission', async () => {
    mockGets([activeJob, completedJob], {
      ...systemStatus,
      worker: { ...systemStatus.worker, state: 'busy', current_job_id: activeJob.id },
      queue: { queued: 0, processing: 1, outstanding: 1 },
    })

    render(<App />)

    expect((await screen.findAllByText('JOB active-j')).length).toBeGreaterThan(0)
    expect(screen.getByText('JOB done-job')).toBeInTheDocument()
    expect(screen.getByText('step 40 · 2 min remaining')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /work already queued or processing/i })).toBeDisabled()
    expect(screen.getByText('Processing job active-j')).toBeInTheDocument()
    expect(screen.getByText('AMD Radeon RX 9070 XT')).toBeInTheDocument()
    expect(window.localStorage.getItem('refsr-selected-job')).toBe(activeJob.id)
  })

  it('submits reference-only mode without frame analysis', async () => {
    const queued = {
      ...activeJob, id: 'reference-only', state: 'queued', stage: 'queued', progress: 0,
      message: 'Queued for reference-only processing', matching_mode: 'reference_only',
    }
    axios.post.mockResolvedValue({ data: queued })
    const { container } = render(<App />)
    const [lowInput, referenceInput] = container.querySelectorAll('input[type="file"]')
    fireEvent.change(lowInput, { target: { files: [new File(['low'], 'low.mp4', { type: 'video/mp4' })] } })
    fireEvent.change(referenceInput, { target: { files: [new File(['reference'], 'reference.mp4', { type: 'video/mp4' })] } })
    fireEvent.click(screen.getByRole('radio', { name: /skip matching/i }))

    expect(screen.getByRole('button', { name: /start reference-only processing/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /start reference-only processing/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalled())
    const [, data] = axios.post.mock.calls[0]
    expect(data.get('matching_mode')).toBe('reference_only')
    expect(data.get('preset')).toBe('balanced')
  })

  it('restores the selected historical job from local storage', async () => {
    window.localStorage.setItem('refsr-selected-job', completedJob.id)
    mockGets([activeJob, completedJob])

    render(<App />)

    expect(await screen.findByText('Upscaled video is ready')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /delete job and files/i })).toBeInTheDocument()
  })

  it('bulk cancels active jobs after confirmation', async () => {
    const cancelled = { ...activeJob, state: 'cancelled', stage: 'cancelled', message: 'Job cancelled' }
    mockGets([activeJob])
    axios.post.mockResolvedValue({ data: { affected: 1, jobs: [cancelled] } })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: /cancel all active jobs/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalledWith(expect.stringMatching(/jobs\/cancel-all$/)))
    expect(await screen.findByText(/Job cancelled/)).toBeInTheDocument()
  })

  it('keeps the last loaded list when polling fails', async () => {
    let jobRequests = 0
    axios.get.mockImplementation((url) => {
      if (url.endsWith('/system/status')) return Promise.resolve({ data: systemStatus })
      jobRequests += 1
      return jobRequests === 1 ? Promise.resolve({ data: { jobs: [activeJob] } }) : Promise.reject(new Error('offline'))
    })
    render(<App />)
    expect((await screen.findAllByText('JOB active-j')).length).toBeGreaterThan(0)

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('offline'), { timeout: 2500 })
    expect(screen.getAllByText('JOB active-j').length).toBeGreaterThan(0)
  })

  it('explains when the backend is unreachable while retaining jobs', async () => {
    axios.get.mockImplementation((url) => url.endsWith('/system/status')
      ? Promise.reject(new Error('connection refused'))
      : Promise.resolve({ data: { jobs: [activeJob] } }))

    render(<App />)

    expect(await screen.findByText('Unreachable')).toBeInTheDocument()
    expect(screen.getByText(/no worker can claim queued work/i)).toBeInTheDocument()
    expect(screen.getAllByText('JOB active-j').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /recheck gpu/i })).toBeDisabled()
  })

  it('shows a GPU failure and starts a manual recheck', async () => {
    const unavailable = {
      ...systemStatus,
      gpu: { ...systemStatus.gpu, state: 'unavailable', name: null, error: 'RuntimeError: driver missing' },
    }
    mockGets([], unavailable)
    axios.post.mockResolvedValue({ data: { ...unavailable.gpu, state: 'checking', error: null } })
    render(<App />)

    expect(await screen.findByText(/GPU check failed: RuntimeError: driver missing/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /recheck gpu/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalledWith(expect.stringMatching(/system\/gpu\/recheck$/)))
    expect(screen.getByText('Checking availability')).toBeInTheDocument()
  })

  it('requires boundary review before processing a proposed segment', async () => {
    const reviewJob = {
      ...activeJob, state: 'awaiting_match_review', stage: 'awaiting_match_review', progress: .07,
      message: 'Review proposed frame matches', needs_review: true,
    }
    const segment = {
      id: 'segment-1', confidence: .9, origin: 'automatic', status: 'proposed',
      low_start: { frame_index: 10, pts: 10, time_seconds: 1 },
      low_end: { frame_index: 50, pts: 50, time_seconds: 5 },
      reference_start: { frame_index: 8, pts: 8, time_seconds: 1 },
      reference_end: { frame_index: 40, pts: 40, time_seconds: 5 },
    }
    const review = {
      revision: 1, segments: [segment], summary: { proposed_segments: 1, matched_seconds: 4 },
      media: {
        low: { fps: 10, frame_count: 100, duration: 10 },
        reference: { fps: 8, frame_count: 80, duration: 10 },
      },
      proxy_urls: { low: '/api/v1/jobs/review/navigation/low', reference: '/api/v1/jobs/review/navigation/reference' },
    }
    axios.get.mockImplementation((url) => {
      if (url.endsWith('/system/status')) return Promise.resolve({ data: systemStatus })
      if (url.endsWith('/match-review')) return Promise.resolve({ data: review })
      return Promise.resolve({ data: { jobs: [reviewJob] } })
    })
    axios.put.mockResolvedValue({ data: {
      ...review, revision: 2, segments: [{ ...segment, status: 'confirmed' }],
      summary: { confirmed_segments: 1, proposed_segments: 0, matched_seconds: 4 },
    } })

    render(<App />)

    expect(await screen.findByText('Start frames')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /approve matches/i })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: /confirm boundaries/i }))

    await waitFor(() => expect(axios.put).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/), expect.objectContaining({ revision: 1 }),
    ))
    expect(await screen.findByRole('button', { name: /approve matches/i })).toBeEnabled()
  })
})
