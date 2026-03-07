# Decision: Browser Support Check Location and Pattern

**Author:** Togusa  
**Date:** 2026-03-07  
**Issue:** #26

## Decision

`detectBrowserSupport()` is called **at module load time** (top-level constant in App.tsx, outside the component function), not inside a `useEffect` or `useState`. This is intentional.

## Rationale

- The check is synchronous and cheap — no async work, no side effects.
- Running it at module load means the result is available on the first render, so the error UI appears immediately with no flash of unsupported content.
- A `useEffect`-based approach would require a loading state, causing a brief flash where the normal UI appears before the gate renders.
- The constant is module-scoped, so tests that need to override it must mock the `detectBrowserSupport` export from `../inference/browserSupport` (using `vi.spyOn` or `vi.mock`).

## Consequence

- App.tsx imports `detectBrowserSupport` from `browserSupport.ts`; this import must not be removed or renamed without updating the gate.
- Test files for App-level behaviour must mock the module, not the DOM APIs, to simulate an unsupported browser.
- The `BrowserUnsupported` component is a pure display component — it receives the full `BrowserSupportResult` as a prop; it does not call `detectBrowserSupport()` itself.

## Implementation (PR #32)

- **Status:** IMPLEMENTED → PR #32 → dev
- **Files:** `App.tsx` (gate + guard on model fetch), `components/BrowserUnsupported.tsx` (error UI), `components/__tests__/BrowserUnsupported.test.tsx` (4 unit tests)
- **Test environment note:** React component tests require `// @vitest-environment jsdom` per-file directive and explicit `cleanup()` from `@testing-library/react` in `afterEach`. The project does not configure jsdom globally or auto-cleanup.
- **Assertion pattern:** Use `container.textContent` rather than `screen.getByText()` when the target text spans nested elements (e.g. `<strong>` inside `<p>`) to avoid "multiple elements found" errors.
