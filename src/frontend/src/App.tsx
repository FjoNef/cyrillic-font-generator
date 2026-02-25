import { useCallback } from 'react';
import { useAppStore } from './stores/appStore';
import FontUpload from './components/FontUpload';
import ModelLoadingBar from './components/ModelLoadingBar';
import GlyphPreview from './components/GlyphPreview';

export default function App() {
  const { uploadedFont, fontName, modelStatus, generationStatus, generatedGlyphs } = useAppStore();

  const canGenerate = uploadedFont !== null && modelStatus === 'ready';
  const hasResults = generatedGlyphs.size > 0;

  const handleGenerate = useCallback(() => {
    // TODO: trigger inference pipeline — OnnxInference + FontLoader
    console.log('Generate clicked — inference pipeline not yet wired');
  }, []);

  const handleDownload = useCallback(() => {
    // TODO: assemble font via FontLoader.assembleCyrillicFont, trigger download
    console.log('Download clicked — font assembly not yet wired');
  }, []);

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
            {generationStatus === 'running' ? 'Generating…' : 'Generate'}
          </button>
          {modelStatus === 'idle' && !uploadedFont && (
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
            <button
              onClick={handleDownload}
              className="px-6 py-2 bg-green-600 text-white rounded-lg font-medium
                         hover:bg-green-700 transition-colors"
            >
              Download .otf
            </button>
          </section>
        )}
      </main>
    </div>
  );
}
