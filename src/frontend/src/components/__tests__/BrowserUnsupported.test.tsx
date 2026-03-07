// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import BrowserUnsupported from '../BrowserUnsupported';
import type { BrowserSupportResult } from '../../inference/browserSupport';

// Mock detectBrowserSupport so App.tsx can be tested without live DOM APIs.
vi.mock('../../inference/browserSupport', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../inference/browserSupport')>();
  return { ...actual };
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function makeSupport(overrides: Partial<BrowserSupportResult>): BrowserSupportResult {
  return {
    supported: false,
    hasWasm: true,
    hasWorkers: true,
    hasSharedArrayBuffer: true,
    hasWebGL: true,
    executionProviders: ['webgl', 'wasm'],
    ...overrides,
  };
}

describe('BrowserUnsupported', () => {
  it('renders when WebAssembly is missing', () => {
    const support = makeSupport({ supported: false, hasWasm: false });
    const { container } = render(<BrowserUnsupported support={support} />);

    expect(screen.getByTestId('browser-unsupported')).toBeTruthy();
    expect(container.textContent).toMatch(/WebAssembly.*not available/i);
  });

  it('renders when Web Workers are missing', () => {
    const support = makeSupport({ supported: false, hasWorkers: false });
    const { container } = render(<BrowserUnsupported support={support} />);

    expect(screen.getByTestId('browser-unsupported')).toBeTruthy();
    expect(container.textContent).toMatch(/Web Workers.*not available/i);
  });

  it('lists all missing capabilities when both are absent', () => {
    const support = makeSupport({ supported: false, hasWasm: false, hasWorkers: false });
    const { container } = render(<BrowserUnsupported support={support} />);

    expect(container.textContent).toMatch(/WebAssembly.*not available/i);
    expect(container.textContent).toMatch(/Web Workers.*not available/i);
  });

  it('includes compatible browser suggestion', () => {
    const support = makeSupport({ supported: false, hasWasm: false });
    const { container } = render(<BrowserUnsupported support={support} />);

    const text = container.textContent ?? '';
    expect(text).toMatch(/Chrome 90\+/i);
    expect(text).toMatch(/Firefox 90\+/i);
    expect(text).toMatch(/Edge 90\+/i);
  });
});
