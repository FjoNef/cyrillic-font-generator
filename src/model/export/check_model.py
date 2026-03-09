"""
check_model.py — Fast sanity check for an exported ONNX model.

Runs entirely in Python using onnxruntime.  No fonts, no training data.
Completes in < 10 seconds on any hardware.

Usage
-----
    python export/check_model.py models/v1/generator.onnx

    # With optional regression baseline folder:
    python export/check_model.py models/v1/generator.onnx --baselines test_outputs/

Exit codes
----------
    0  All checks passed.
    1  One or more checks failed.

Convention reminder
-------------------
    Model output space:   +1.0 = black ink,  -1.0 = white background.
    Postprocessing:       pixel = ((1 - output) / 2) * 255
    → +1.0 → pixel 0   (black / ink)
    → -1.0 → pixel 255 (white / background)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH          = 1
N_STYLE        = 10   # number of style reference glyphs
HEIGHT         = 128
WIDTH          = 128
N_CHARS        = 66   # number of Cyrillic target characters (0-indexed)

# Thresholds
RANGE_EPSILON  = 0.1   # small tolerance for INT8 quantisation artefacts
INK_THRESHOLD  = 0.0   # output values above this are considered ink pixels
INK_MIN_FRAC   = 0.01  # at least 1 % of pixels must be ink
STYLE_MAD_MIN  = 0.01  # minimum MAD between contrasting style runs
CHAR_MAD_MIN   = 0.005 # minimum MAD between two different characters
REGR_MAD_MAX   = 0.1   # maximum drift vs saved baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def _run(sess: ort.InferenceSession,
         style_fill: float,
         char_idx: int = 0) -> np.ndarray:
    """Single inference call.  Returns output [1, 1, H, W]."""
    style = np.full(
        (BATCH, N_STYLE, 1, HEIGHT, WIDTH), fill_value=style_fill, dtype=np.float32
    )
    char = np.array([char_idx], dtype=np.int64)
    outputs = sess.run(None, {"style_glyphs": style, "char_index": char})
    return outputs[0]  # [B, 1, H, W]


def _pass(label: str, value: str) -> None:
    print(f"  ✅  PASS  {label:<45}  {value}")


def _fail(label: str, value: str) -> None:
    print(f"  ❌  FAIL  {label:<45}  {value}")


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_output_range(output: np.ndarray) -> bool:
    lo, hi = float(output.min()), float(output.max())
    passed = lo >= (-1.0 - RANGE_EPSILON) and hi <= (1.0 + RANGE_EPSILON)
    label = "Output range in [-1, 1]"
    value = f"min={lo:.4f}  max={hi:.4f}  (epsilon={RANGE_EPSILON})"
    (_pass if passed else _fail)(label, value)
    return passed


def check_non_blank(output: np.ndarray) -> bool:
    """At least INK_MIN_FRAC of pixels must be above INK_THRESHOLD.

    In model space: +1.0 = black ink, -1.0 = white background.
    A blank (all-white) glyph would have all values near -1.0 — zero pixels
    above INK_THRESHOLD → fails.
    """
    ink_frac = float(np.mean(output > INK_THRESHOLD))
    passed = ink_frac >= INK_MIN_FRAC
    label = f"Non-blank (≥{INK_MIN_FRAC*100:.0f}% ink pixels)"
    value = f"ink_frac={ink_frac:.4f}  (threshold output>{INK_THRESHOLD})"
    (_pass if passed else _fail)(label, value)
    return passed


def check_style_conditioning(sess: ort.InferenceSession) -> bool:
    """Maximally different style inputs must produce different outputs (MAD > STYLE_MAD_MIN)."""
    out_white = _run(sess, style_fill=+1.0)   # all-ink style
    out_black = _run(sess, style_fill=-1.0)   # all-background style
    mad = float(np.mean(np.abs(out_white - out_black)))
    passed = mad > STYLE_MAD_MIN
    label = f"Style conditioning (MAD > {STYLE_MAD_MIN})"
    value = f"MAD={mad:.6f}  (white-fill vs black-fill style)"
    (_pass if passed else _fail)(label, value)
    return passed


def check_char_isolation(sess: ort.InferenceSession) -> bool:
    """Different char_index values should produce different outputs."""
    out_0 = _run(sess, style_fill=0.0, char_idx=0)
    out_1 = _run(sess, style_fill=0.0, char_idx=min(1, N_CHARS - 1))
    out_max = _run(sess, style_fill=0.0, char_idx=N_CHARS - 1)
    mad_01 = float(np.mean(np.abs(out_0 - out_1)))
    mad_0max = float(np.mean(np.abs(out_0 - out_max)))
    passed = mad_01 > CHAR_MAD_MIN or mad_0max > CHAR_MAD_MIN
    label = f"Char isolation (MAD > {CHAR_MAD_MIN})"
    value = f"MAD(0,1)={mad_01:.6f}  MAD(0,{N_CHARS-1})={mad_0max:.6f}"
    (_pass if passed else _fail)(label, value)
    return passed


def check_regression(sess: ort.InferenceSession, baselines_dir: Path) -> bool | None:
    """Compare current outputs against saved .npy baselines.  Returns None if no baselines."""
    baseline_files = list(baselines_dir.glob("*.npy"))
    if not baseline_files:
        print(f"  ⏭️   SKIP  Regression baseline                              "
              f"  (no .npy files in {baselines_dir})")
        return None

    total, failures = 0, 0
    for bf in sorted(baseline_files):
        # Filename convention: style{fill}_char{idx}.npy
        # Parse or use defaults for unknown files.
        parts = bf.stem.split("_")
        try:
            style_fill = float(parts[0].replace("style", ""))
            char_idx   = int(parts[1].replace("char", ""))
        except Exception:
            style_fill, char_idx = 0.0, 0

        baseline = np.load(str(bf))
        current  = _run(sess, style_fill=style_fill, char_idx=char_idx)
        mad      = float(np.mean(np.abs(current - baseline)))
        ok       = mad <= REGR_MAD_MAX
        total += 1
        if not ok:
            failures += 1
        tag = "✅" if ok else "❌"
        print(f"  {tag}  {'PASS' if ok else 'FAIL'}  "
              f"Regression {bf.name:<35}  MAD={mad:.6f}  (limit={REGR_MAD_MAX})")

    passed = failures == 0
    overall = "PASS" if passed else f"FAIL  ({failures}/{total} drifted)"
    print(f"  {'✅' if passed else '❌'}  {'PASS' if passed else 'FAIL'}  "
          f"{'Regression overall':<45}  {overall}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(model_path: str, baselines_dir: str | None = None) -> bool:
    print(f"\n{'='*70}")
    print(f"  Model Sanity Check — {model_path}")
    print(f"{'='*70}\n")

    # Load model.
    print("  Loading model…")
    try:
        sess = _session(model_path)
    except Exception as e:
        print(f"  ❌  FAIL  Could not load model: {e}")
        return False

    inputs = {inp.name: inp.shape for inp in sess.get_inputs()}
    outputs_meta = {out.name: out.shape for out in sess.get_outputs()}
    print(f"  Inputs:  {inputs}")
    print(f"  Outputs: {outputs_meta}\n")

    # Run a neutral inference for range + non-blank checks (grey style, char 0).
    neutral_out = _run(sess, style_fill=0.0, char_idx=0)

    results: list[bool] = []

    # --- Check 1: Output range ---
    results.append(check_output_range(neutral_out))

    # --- Check 2: Non-blank ---
    results.append(check_non_blank(neutral_out))

    # --- Check 3: Style conditioning ---
    results.append(check_style_conditioning(sess))

    # --- Check 4: Character isolation ---
    results.append(check_char_isolation(sess))

    # --- Check 5: Regression baseline (optional) ---
    if baselines_dir:
        bp = Path(baselines_dir)
        if bp.is_dir():
            reg_result = check_regression(sess, bp)
            if reg_result is not None:
                results.append(reg_result)
        else:
            print(f"  ⚠️   WARN  Baselines dir not found: {baselines_dir}")

    # --- Summary ---
    all_passed = all(results)
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*70}")
    if all_passed:
        print(f"  ✅  ALL CHECKS PASSED ({n_pass}/{n_total})")
    else:
        print(f"  ❌  CHECKS FAILED ({n_total - n_pass} of {n_total} failed)")
    print(f"{'='*70}\n")

    return all_passed


# ---------------------------------------------------------------------------
# Baseline save utility
# ---------------------------------------------------------------------------

def save_baselines(model_path: str, out_dir: str) -> None:
    """Save reference outputs for future regression checks.

    Saves outputs for a representative set of (style_fill, char_idx) pairs.
    """
    baselines = [
        (0.0,  0),
        (0.0, 32),
        (0.0, 65),
        (1.0,  0),
        (-1.0, 0),
    ]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sess = _session(model_path)
    for fill, cidx in baselines:
        output = _run(sess, style_fill=fill, char_idx=cidx)
        fname = out_path / f"style{fill:+.1f}_char{cidx:03d}.npy"
        np.save(str(fname), output)
        print(f"  Saved baseline: {fname}")
    print(f"\n✅ Saved {len(baselines)} baselines to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast ONNX model sanity check for the Cyrillic font generator."
    )
    parser.add_argument(
        "model",
        help="Path to the ONNX model file (e.g. models/v1/generator.onnx).",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help="Directory containing .npy regression baselines (optional).",
    )
    parser.add_argument(
        "--save-baselines",
        metavar="DIR",
        default=None,
        help="Save reference outputs to DIR (use after confirming a good model).",
    )
    args = parser.parse_args()

    if args.save_baselines:
        save_baselines(args.model, args.save_baselines)
        sys.exit(0)

    ok = run_checks(args.model, baselines_dir=args.baselines)
    sys.exit(0 if ok else 1)
