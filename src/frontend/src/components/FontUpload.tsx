import { useCallback, useRef } from 'react';
import { useAppStore } from '../stores/appStore';
import { FontLoader } from '../font/FontLoader';

const ACCEPTED_TYPES = ['.otf', '.ttf', '.woff', '.woff2'];

export default function FontUpload() {
  const { setUploadedFont } = useAppStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const loader = useRef(new FontLoader());

  const processFile = useCallback(
    async (file: File) => {
      const buffer = await file.arrayBuffer();
      // Validate by parsing — throws if invalid
      await loader.current.loadFont(buffer);
      setUploadedFont(buffer, file.name);
    },
    [setUploadedFont]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
      className="border-2 border-dashed border-gray-300 rounded-xl p-10
                 flex flex-col items-center justify-center gap-2 cursor-pointer
                 hover:border-blue-400 hover:bg-blue-50 transition-colors"
    >
      <span className="text-4xl">🔤</span>
      <p className="font-medium text-gray-700">Drag & drop a font file here</p>
      <p className="text-sm text-gray-400">
        or click to browse — {ACCEPTED_TYPES.join(', ')}
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
}
