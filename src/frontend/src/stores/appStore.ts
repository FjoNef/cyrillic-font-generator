import { create } from 'zustand';

type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';
type GenerationStatus = 'idle' | 'running' | 'done' | 'error';

interface AppState {
  // Font
  uploadedFont: ArrayBuffer | null;
  fontName: string | null;
  // Model
  modelStatus: ModelStatus;
  modelLoadProgress: number; // 0-100
  // Generation
  generationStatus: GenerationStatus;
  generatedGlyphs: Map<string, ImageData>; // char → glyph image
  // Actions
  setUploadedFont: (buffer: ArrayBuffer, name: string) => void;
  setModelStatus: (status: ModelStatus, progress?: number) => void;
  setGenerationStatus: (status: GenerationStatus) => void;
  setGeneratedGlyph: (char: string, glyph: ImageData) => void;
}

export const useAppStore = create<AppState>((set) => ({
  uploadedFont: null,
  fontName: null,
  modelStatus: 'idle',
  modelLoadProgress: 0,
  generationStatus: 'idle',
  generatedGlyphs: new Map(),

  setUploadedFont: (buffer, name) =>
    set({ uploadedFont: buffer, fontName: name }),

  setModelStatus: (status, progress) =>
    set((state) => ({
      modelStatus: status,
      modelLoadProgress: progress ?? state.modelLoadProgress,
    })),

  setGenerationStatus: (status) =>
    set({ generationStatus: status }),

  setGeneratedGlyph: (char, glyph) =>
    set((state) => {
      const next = new Map(state.generatedGlyphs);
      next.set(char, glyph);
      return { generatedGlyphs: next };
    }),
}));
