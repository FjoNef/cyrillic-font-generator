/**
 * Triggers a browser download of a font file.
 *
 * Creates a temporary Blob URL, simulates an anchor click, then cleans up.
 *
 * @param fontBuffer  ArrayBuffer containing the binary font data (OTF/TTF)
 * @param filename    Download filename; defaults to "generated-cyrillic.otf"
 */
export function downloadFont(
  fontBuffer: ArrayBuffer,
  filename: string = 'generated-cyrillic.otf'
): void {
  const blob = new Blob([fontBuffer], { type: 'font/otf' });
  const url  = URL.createObjectURL(blob);

  const anchor = document.createElement('a');
  anchor.href     = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  // Release the object URL after the click event has been dispatched
  URL.revokeObjectURL(url);
}
