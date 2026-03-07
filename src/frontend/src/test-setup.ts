// Define browser APIs not implemented by jsdom
if (typeof URL.createObjectURL === 'undefined') {
  Object.defineProperty(URL, 'createObjectURL', { writable: true, configurable: true, value: () => '' });
}
if (typeof URL.revokeObjectURL === 'undefined') {
  Object.defineProperty(URL, 'revokeObjectURL', { writable: true, configurable: true, value: () => {} });
}

// Provide a minimal canvas 2D context mock for jsdom (no native canvas bindings).
// Only applied when running in a browser-like environment (jsdom/happy-dom).
if (typeof HTMLCanvasElement !== 'undefined') {
  const mock2dContext = {
    clearRect: () => {},
    fillRect: () => {},
    fill: () => {},
    set fillStyle(_: string) {},
    get fillStyle() { return '#000'; },
    getImageData: (_x: number, _y: number, w: number, h: number) => ({
      data: new Uint8ClampedArray(w * h * 4).fill(255), // all-white pixels
      width: w,
      height: h,
    }),
  };

  HTMLCanvasElement.prototype.getContext = function (contextId: string) {
    if (contextId === '2d') return mock2dContext as unknown as CanvasRenderingContext2D;
    return null;
  } as typeof HTMLCanvasElement.prototype.getContext;
}

// Path2D is not available in jsdom; provide a no-op stub.
if (typeof (globalThis as any).Path2D === 'undefined') {
  (globalThis as any).Path2D = class Path2D {
    constructor(_?: string) {}
  };
}
