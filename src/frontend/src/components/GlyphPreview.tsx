import { useEffect, useRef } from 'react';
import { CYRILLIC_CHARS } from '../font/cyrillicCharset';

interface GlyphPreviewProps {
  glyphs: Map<string, ImageData>;
}

/** Renders a single ImageData onto a small canvas. */
function GlyphCell({ char, imageData }: { char: string; imageData: ImageData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(imageData, 0, 0);
  }, [imageData]);

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        width={128}
        height={128}
        className="border border-gray-200 rounded"
        style={{ width: 64, height: 64 }}
        title={`U+${char.codePointAt(0)!.toString(16).toUpperCase().padStart(4, '0')}`}
      />
      <span className="text-sm font-mono text-gray-700">{char}</span>
    </div>
  );
}

/** Grid of all generated Cyrillic glyphs. */
export default function GlyphPreview({ glyphs }: GlyphPreviewProps) {
  const uppercase = CYRILLIC_CHARS.filter((c) => c.isUppercase);
  const lowercase = CYRILLIC_CHARS.filter((c) => !c.isUppercase);

  const renderRow = (chars: typeof CYRILLIC_CHARS) =>
    chars.map(({ char }) => {
      const imageData = glyphs.get(char);
      if (!imageData) return null;
      return <GlyphCell key={char} char={char} imageData={imageData} />;
    });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">{renderRow(uppercase)}</div>
      <div className="flex flex-wrap gap-3">{renderRow(lowercase)}</div>
    </div>
  );
}
