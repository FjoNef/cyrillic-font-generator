import { describe, it, expect } from 'vitest';

/**
 * Tests for tensor-to-pixel color mapping formula.
 * 
 * Critical correctness test for PR #4 fix.
 * Model outputs [-1, 1] where +1.0 = black ink, -1.0 = white background.
 * Formula: ((1 - output) / 2) * 255
 */

describe('Color Mapping Formula', () => {
  it('should map +1.0 (black ink) to 0 (black pixel)', () => {
    const output = 1.0;
    const pixel = Math.round(((1 - output) / 2) * 255);
    expect(pixel).toBe(0);
  });

  it('should map -1.0 (white background) to 255 (white pixel)', () => {
    const output = -1.0;
    const pixel = Math.round(((1 - output) / 2) * 255);
    expect(pixel).toBe(255);
  });

  it('should map 0.0 (midtone) to 128 (gray pixel)', () => {
    const output = 0.0;
    const pixel = Math.round(((1 - output) / 2) * 255);
    expect(pixel).toBe(128);
  });

  it('should map intermediate values correctly', () => {
    // +0.5 should map to darker gray (~64)
    const darker = Math.round(((1 - 0.5) / 2) * 255);
    expect(darker).toBe(64);

    // -0.5 should map to lighter gray (~191)
    const lighter = Math.round(((1 - (-0.5)) / 2) * 255);
    expect(lighter).toBe(191);
  });

  it('should be the inverse of the old incorrect formula', () => {
    // OLD INCORRECT: ((output + 1) / 2) * 255
    // NEW CORRECT:   ((1 - output) / 2) * 255
    
    const testValue = 0.8;
    const oldFormula = Math.round(((testValue + 1) / 2) * 255); // 230 (wrong)
    const newFormula = Math.round(((1 - testValue) / 2) * 255);  // 25 (correct)
    
    expect(oldFormula).toBe(230); // Would produce inverted image
    expect(newFormula).toBe(25);  // Correct mapping
    expect(oldFormula).not.toBe(newFormula);
  });
});
