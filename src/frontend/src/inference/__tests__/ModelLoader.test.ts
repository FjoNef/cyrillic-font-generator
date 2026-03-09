import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ModelLoader } from '../ModelLoader';

/**
 * Tests for ModelLoader singleton.
 * 
 * Coverage:
 * - Model load with progress tracking
 * - Concurrent load requests return same promise
 * - Inference request/response flow with request IDs
 * - Error handling: network failure, worker error, inference error
 */

describe('ModelLoader', () => {
  let mockWorker: any;
  let modelLoader: ModelLoader;

  beforeEach(() => {
    // Mock Worker constructor
    mockWorker = {
      postMessage: vi.fn(),
      terminate: vi.fn(),
      onmessage: null,
      onerror: null,
    };

    // Capture message handler
    global.Worker = vi.fn().mockImplementation(() => {
      return mockWorker;
    }) as any;

    // Fresh singleton per test to avoid stale loadPromise/worker state
    modelLoader = new ModelLoader();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should load model and report progress', async () => {
    const progressCallback = vi.fn();
    const loadPromise = modelLoader.load('/api/model', progressCallback);

    // Worker should be created
    expect(global.Worker).toHaveBeenCalledWith(
      expect.any(URL),
      expect.objectContaining({ type: 'module' })
    );

    // Worker should receive load message
    expect(mockWorker.postMessage).toHaveBeenCalledWith({
      type: 'load',
      modelUrl: '/api/model',
    });

    // Simulate progress updates
    mockWorker.onmessage({ data: { type: 'progress', progress: 25 } });
    mockWorker.onmessage({ data: { type: 'progress', progress: 50 } });
    mockWorker.onmessage({ data: { type: 'progress', progress: 100 } });

    expect(progressCallback).toHaveBeenCalledWith(25);
    expect(progressCallback).toHaveBeenCalledWith(50);
    expect(progressCallback).toHaveBeenCalledWith(100);

    // Complete load
    mockWorker.onmessage({ data: { type: 'loaded' } });

    await loadPromise;
    expect(modelLoader.isReady()).toBe(true);
  });

  it('should return same promise for concurrent load calls', async () => {
    const promise1 = modelLoader.load('/api/model');
    const promise2 = modelLoader.load('/api/model');

    expect(promise1).toBe(promise2);

    // Complete load
    mockWorker.onmessage({ data: { type: 'loaded' } });

    await Promise.all([promise1, promise2]);
  });

  it('should handle load errors', async () => {
    const loadPromise = modelLoader.load('/api/model');

    // Simulate error
    mockWorker.onmessage({ data: { type: 'error', message: 'Network failure' } });

    await expect(loadPromise).rejects.toThrow('Network failure');
  });

  it('should handle worker error events', async () => {
    const loadPromise = modelLoader.load('/api/model');

    // Trigger onerror with an ErrorEvent (has message/filename/lineno)
    const errEvent = new ErrorEvent('error', {
      message: 'Worker crashed',
      filename: 'worker.js',
      lineno: 42,
      colno: 1,
    });
    mockWorker.onerror(errEvent);

    await expect(loadPromise).rejects.toThrow('Worker error: Worker crashed');
  });

  it('should run inference with request ID tracking', async () => {
    // Load model first
    const loadPromise = modelLoader.load('/api/model');
    mockWorker.onmessage({ data: { type: 'loaded' } });
    await loadPromise;

    // Create synthetic style glyphs (10 * 128 * 128 = 163840 floats)
    const styleGlyphs = new Float32Array(163840).fill(0.5);
    const charIndex = 5;

    const inferPromise = modelLoader.infer(styleGlyphs, charIndex);

    // Flush microtasks so the await inside infer() completes and postMessage is called
    await Promise.resolve();

    // Worker should receive infer message with request ID
    expect(mockWorker.postMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'infer',
        styleGlyphs,
        charIndex,
        requestId: expect.stringMatching(/^req-\d+$/),
      })
    );

    // Get the request ID from the call
    const inferCall = mockWorker.postMessage.mock.calls.find(
      (call: any) => call[0].type === 'infer'
    );
    const requestId = inferCall[0].requestId;

    // Simulate inference result
    const output = new Float32Array(16384).fill(0.8);
    mockWorker.onmessage({ data: { type: 'result', output, requestId } });

    const result = await inferPromise;
    expect(result).toEqual(output);
  });

  it('should handle concurrent inference requests', async () => {
    // Load model
    const loadPromise = modelLoader.load('/api/model');
    mockWorker.onmessage({ data: { type: 'loaded' } });
    await loadPromise;

    const styleGlyphs = new Float32Array(163840);
    const infer1 = modelLoader.infer(styleGlyphs, 0);
    const infer2 = modelLoader.infer(styleGlyphs, 1);

    // Flush microtasks so postMessage calls are made
    await Promise.resolve();

    // Extract request IDs
    const calls = mockWorker.postMessage.mock.calls.filter(
      (call: any) => call[0].type === 'infer'
    );
    const req1 = calls[0][0].requestId;
    const req2 = calls[1][0].requestId;

    expect(req1).not.toBe(req2);

    // Respond out of order (req2 first)
    const output2 = new Float32Array(16384).fill(0.9);
    mockWorker.onmessage({ data: { type: 'result', output: output2, requestId: req2 } });

    const output1 = new Float32Array(16384).fill(0.1);
    mockWorker.onmessage({ data: { type: 'result', output: output1, requestId: req1 } });

    const [result1, result2] = await Promise.all([infer1, infer2]);
    expect(result1).toEqual(output1);
    expect(result2).toEqual(output2);
  });

  it('should reject inference if model not loaded', async () => {
    const styleGlyphs = new Float32Array(163840);
    await expect(modelLoader.infer(styleGlyphs, 0)).rejects.toThrow('Model not loaded');
  });

  it('should handle inference errors', async () => {
    // Load model
    const loadPromise = modelLoader.load('/api/model');
    mockWorker.onmessage({ data: { type: 'loaded' } });
    await loadPromise;

    const styleGlyphs = new Float32Array(163840);
    const inferPromise = modelLoader.infer(styleGlyphs, 0);

    // Flush microtasks so postMessage is called before we read mock.calls
    await Promise.resolve();

    // Get request ID
    const inferCall = mockWorker.postMessage.mock.calls.find(
      (call: any) => call[0].type === 'infer'
    );
    const requestId = inferCall[0].requestId;

    // Simulate error
    mockWorker.onmessage({
      data: { type: 'error', message: 'ONNX Runtime error', requestId },
    });

    await expect(inferPromise).rejects.toThrow('ONNX Runtime error');
  });
});
