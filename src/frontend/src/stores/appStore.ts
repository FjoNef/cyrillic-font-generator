import { create } from 'zustand';

type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';
type GenerationStatus = 'idle' | 'running' | 'done' | 'error';

interface AppState {
  // Font
  uploadedFont: ArrayBuffer | null;
  fontName: string | null;
  styleGlyphs: Float32Array | null;
  // Model
  modelStatus: ModelStatus;
  modelLoadProgress: number; // 0-100
  // Generation
  generationStatus: GenerationStatus;
  generationProgress: number; // 0-66
  generatedGlyphs: Map<string, ImageData>; // char → glyph image
  fontBuffer: ArrayBuffer | null;
  // Actions
  setUploadedFont: (buffer: ArrayBuffer, name: string) => void;
  setStyleGlyphs: (glyphs: Float32Array) => void;
  setModelStatus: (status: ModelStatus, progress?: number) => void;
  setGenerationStatus: (status: GenerationStatus) => void;
  setGenerationProgress: (progress: number) => void;
  setGeneratedGlyph: (char: string, glyph: ImageData) => void;
  setFontBuffer: (buffer: ArrayBuffer) => void;
  reset: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  uploadedFont: null,
  fontName: null,
  styleGlyphs: null,
  modelStatus: 'idle',
  modelLoadProgress: 0,
  generationStatus: 'idle',
  generationProgress: 0,
  generatedGlyphs: new Map(),
  fontBuffer: null,

  setUploadedFont: (buffer, name) =>
    set({ uploadedFont: buffer, fontName: name }),

  setStyleGlyphs: (glyphs) =>
    set({ styleGlyphs: glyphs }),

  setModelStatus: (status, progress) =>
    set((state) => ({
      modelStatus: status,
      modelLoadProgress: progress ?? state.modelLoadProgress,
    })),

  setGenerationStatus: (status) =>
    set({ generationStatus: status }),

  setGenerationProgress: (progress) =>
    set({ generationProgress: progress }),

  setGeneratedGlyph: (char, glyph) =>
    set((state) => {
      const next = new Map(state.generatedGlyphs);
      next.set(char, glyph);
      return { generatedGlyphs: next };
    }),

  setFontBuffer: (buffer) =>
    set({ fontBuffer: buffer }),

  reset: () =>
    set({
      generationStatus: 'idle',
      generationProgress: 0,
      generatedGlyphs: new Map(),
      fontBuffer: null,
    }),
}));
