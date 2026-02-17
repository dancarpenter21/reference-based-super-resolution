import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import App from '../App';

describe('App', () => {
    it('renders without crashing', () => {
        render(<App />);
        // Adjust expectation based on actual App content, for now just checking render
        // If App has specific text, we can look for it e.g.:
        // expect(screen.getByText(/reference-based super-resolution/i)).toBeInTheDocument();
    });
});
