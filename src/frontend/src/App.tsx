import { useCallback, useEffect } from 'react';
import { useAppStore } from './stores/appStore';
import { modelLoader } from './inference/ModelLoader';
import { CYRILLIC_CHARS } from './font/cyrillicCharset';
import { assembleFontFromGlyphs } from './FontAssembler';
import { downloadFont } from './FontDownloader';
import FontUpload from './components/FontUpload';
import ModelLoadingBar from './components/ModelLoadingBar';
import GlyphPreview from './components/GlyphPreview';
import BrowserUnsupported from './components/BrowserUnsupported';
import { detectBrowserSupport } from './inference/browserSupport';

// Evaluated once at module load — synchronous, no React overhead.
const browserSupport = detectBrowserSupport();

export default function App() {
  const { 
    uploadedFont, 
    fontName, 
    styleGlyphs,
    modelStatus, 
    generationStatus, 
    generationProgress,
    generatedGlyphs,
    fontBuffer,
    setModelStatus,
    setGenerationStatus,
    setGenerationProgress,
    setGeneratedGlyph,
    setFontBuffer,
    reset,
  } = useAppStore();

  // Load model on mount — skipped entirely if browser is unsupported.
  useEffect(() => {
    if (!browserSupport.supported) return;

    const loadModel = async () => {
      setModelStatus('loading', 0);
      try {
        // Fetch manifest first to get the versioned URL (contains SHA-256 of model).
        // Using a versioned URL means the browser can safely cache the model bytes,
        // and will automatically re-download when the model is retrained.
        const manifestRes = await fetch('/api/model/manifest');
        if (!manifestRes.ok) throw new Error(`Manifest fetch failed: ${manifestRes.statusText}`);
        const manifest: { downloadUrl: string } = await manifestRes.json();

        // ⚠️ Critical: Extract only the pathname from downloadUrl.
        // The manifest returns an absolute URL (http://localhost:5000/api/model/...),
        // but the worker needs to fetch via the Vite proxy on port 5173.
        // Stripping to pathname ensures the fetch goes through the proxy.
        const url = new URL(manifest.downloadUrl, window.location.origin);
        const modelPath = url.pathname; // e.g. "/api/model/v1/generator.onnx"

        await modelLoader.load(modelPath, (progress) => {
          setModelStatus('loading', progress);
        });
        setModelStatus('ready', 100);
      } catch (error) {
        console.error('Model load failed:', error);
        setModelStatus('error', 0);
      }
    };

    loadModel();
  }, [setModelStatus]);

  const canGenerate = styleGlyphs !== null && modelStatus === 'ready';
  const hasResults = generatedGlyphs.size > 0;

  const handleGenerate = useCallback(async () => {
    if (!styleGlyphs) return;

    reset();
    setGenerationStatus('running');
    setGenerationProgress(0);

    try {
      // Collect raw Float32Array outputs alongside ImageData for preview
      const rawGlyphs = new Map<number, Float32Array>();

      // Generate all 66 Cyrillic glyphs (single inference pass)
      for (let i = 0; i < CYRILLIC_CHARS.length; i++) {
        const { char, index } = CYRILLIC_CHARS[i];

        // Run inference → Float32Array [-1,1]
        const output = await modelLoader.infer(styleGlyphs, index);

        // Store raw output for font assembly
        rawGlyphs.set(index, output);

        // Convert to ImageData for glyph preview
        const pixels = new Uint8ClampedArray(128 * 128 * 4);
        for (let px = 0; px < 128 * 128; px++) {
          const val = Math.round(((1 - output[px]) / 2) * 255);
          pixels[px * 4 + 0] = val; // R
          pixels[px * 4 + 1] = val; // G
          pixels[px * 4 + 2] = val; // B
          pixels[px * 4 + 3] = 255; // A
        }

        const imageData = new ImageData(pixels, 128, 128);
        setGeneratedGlyph(char, imageData);
        setGenerationProgress(i + 1);
      }

      // Assemble OTF from already-generated raw glyphs (no re-inference)
      const buffer = assembleFontFromGlyphs(rawGlyphs, uploadedFont, fontName ?? 'Generated Cyrillic');
      setFontBuffer(buffer);
      setGenerationStatus('done');
    } catch (error) {
      console.error('Generation failed:', error);
      setGenerationStatus('error');
    }
  }, [styleGlyphs, fontName, reset, setGenerationStatus, setGenerationProgress, setGeneratedGlyph, setFontBuffer]);

  const handleDownload = useCallback(() => {
    if (!fontBuffer) return;
    const safeName = (fontName ?? 'generated-cyrillic').replace(/\s+/g, '-').toLowerCase();
    downloadFont(fontBuffer, `${safeName}.otf`);
  }, [fontBuffer, fontName]);

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold tracking-tight">Cyrillic Font Generator</h1>
        <p className="text-sm text-gray-500 mt-1">
          Upload a font, generate Cyrillic glyphs with AI — entirely in your browser.
        </p>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-8">

        {/* Browser capability gate — shown on unsupported browsers */}
        {!browserSupport.supported && (
          <BrowserUnsupported support={browserSupport} />
        )}

        {/* Step 1 — Font Upload */}
        <section>
          <h2 className="text-lg font-semibold mb-3">
            <span className="text-blue-600 mr-2">1.</span>Upload your font
          </h2>
          <FontUpload />
          {fontName && (
            <p className="mt-2 text-sm text-green-600">✓ Loaded: {fontName}</p>
          )}
        </section>

        {/* Model loading progress (shown while loading) */}
        {(modelStatus === 'loading' || modelStatus === 'error') && (
          <ModelLoadingBar />
        )}

        {/* Step 2 — Generate */}
        <section>
          <h2 className="text-lg font-semibold mb-3">
            <span className="text-blue-600 mr-2">2.</span>Generate Cyrillic glyphs
          </h2>
          <button
            onClick={handleGenerate}
            disabled={!canGenerate || generationStatus === 'running'}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium
                       hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            {generationStatus === 'running' ? `Generating Cyrillic glyphs: ${generationProgress}/66…` : 'Generate'}
          </button>
          {!canGenerate && !uploadedFont && (
            <p className="mt-2 text-sm text-gray-400">Upload a font to enable generation.</p>
          )}
          {modelStatus === 'loading' && (
            <p className="mt-2 text-sm text-gray-400">Waiting for AI model to load…</p>
          )}
        </section>

        {/* Step 3 — Preview */}
        {hasResults && (
          <section>
            <h2 className="text-lg font-semibold mb-3">
              <span className="text-blue-600 mr-2">3.</span>Preview
            </h2>
            <GlyphPreview glyphs={generatedGlyphs} />
          </section>
        )}

        {/* Step 4 — Download */}
        {hasResults && (
          <section>
            <h2 className="text-lg font-semibold mb-3">
              <span className="text-blue-600 mr-2">4.</span>Download font
            </h2>
            {generationStatus === 'running' && (
              <p className="mb-2 text-sm text-gray-500">
                Generating glyphs… {generationProgress}/66
              </p>
            )}
            <button
              onClick={handleDownload}
              disabled={generationStatus !== 'done' || !fontBuffer}
              className="px-6 py-2 bg-green-600 text-white rounded-lg font-medium
                         hover:bg-green-700 disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors"
            >
              Download .otf
            </button>
          </section>
        )}
      </main>
    </div>
  );
}
