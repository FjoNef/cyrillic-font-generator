import type { BrowserSupportResult } from '../inference/browserSupport';

interface Props {
  support: BrowserSupportResult;
}

const MISSING_LABELS: Record<string, string> = {
  wasm: 'WebAssembly',
  workers: 'Web Workers',
};

export default function BrowserUnsupported({ support }: Props) {
  const missing: string[] = [];
  if (!support.hasWasm) missing.push(MISSING_LABELS.wasm);
  if (!support.hasWorkers) missing.push(MISSING_LABELS.workers);

  return (
    <div
      role="alert"
      className="rounded-lg border border-amber-300 bg-amber-50 p-5 space-y-3"
      data-testid="browser-unsupported"
    >
      <h2 className="text-base font-semibold text-amber-800">
        Your browser doesn't support this app
      </h2>

      <p className="text-sm text-amber-700">
        The Cyrillic Font Generator runs AI inference entirely in your browser
        and requires features that aren't available here:
      </p>

      <ul className="list-disc list-inside text-sm text-amber-700 space-y-1">
        {missing.map((cap) => (
          <li key={cap}>{cap} — not available</li>
        ))}
      </ul>

      <p className="text-sm text-amber-700">
        <strong>Fix:</strong> Open this page in a modern browser —{' '}
        <strong>Chrome 90+</strong>, <strong>Firefox 90+</strong>, or{' '}
        <strong>Edge 90+</strong> — and make sure it's not running in a
        restricted context (e.g. file://, private browsing with strict
        settings, or certain browser extensions can disable these APIs).
      </p>
    </div>
  );
}
