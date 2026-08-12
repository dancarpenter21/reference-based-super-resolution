import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from '../App'

describe('App', () => {
  it('renders the two-video workflow', () => {
    render(<App />)
    expect(screen.getByText('01 · Complete source')).toBeInTheDocument()
    expect(screen.getByText('02 · Detail reference')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /analyze and upscale/i })).toBeInTheDocument()
  })
})
