# Decision: Style-Conditioning Fix Applied to Training Code

**By:** Major (AI/ML Engineer)  
**Date:** 2026-03-07  
**Status:** ACCEPTED  
**Related diagnosis:** `.squad/decisions/inbox/major-style-conditioning-diagnosis.md`

---

## Context

The model produced identical output regardless of font style. Root-cause diagnosis (logged 2026-03-07)
identified two compounding bugs in `src/model/train/model.py` and `src/model/train/train.py`.

---

## Changes Made

### Bug 1 — UNetGenerator encoder input (`model.py`)

**Before:** `UNetGenerator.forward()` fed `torch.zeros(B, 1, 128, 128, device=device)` into
the encoder. All six U-Net skip connections (e1–e6) were therefore **constant across all inputs**
(they depend only on model weights, not on font data). Style conditioning entered only at the 1×1
bottleneck (`cond_spatial`), then was progressively diluted across 6 decoder stages.

**After:** `UNetGenerator.forward()` now accepts a third argument `style_glyph_0: [B, 1, 128, 128]`
(the first style reference glyph). This tensor is used as the encoder input. All 6 skip connections
now carry real per-font spatial structure at 64×64, 32×32, 16×16, 8×8, 4×4, and 2×2 scales.

**Callers updated:**
- Training loop (`train.py`): passes `cond_glyph` (`style_glyphs[:, 0]`) as the third argument
- ONNX wrapper (`export_onnx.py`): extracts `style_glyphs[:, 0]` and passes it to `generator()`
- ONNX input contract unchanged: `style_glyphs [B, 10, 1, 128, 128]` is still the only image input

### Bug 2 — Loss rebalancing (`model.py`, `train.py`, `train_config.yaml`)

**Before:** `lambda_l1 = 100`, no feature matching. Generator rewarded for pixel-average shapes
with no style differentiation incentive.

**After:**
- `lambda_l1`: 100 → **10** (in `train_config.yaml` and synthetic defaults in `train.py`)
- `lambda_fm = 10` added (`train_config.yaml` + synthetic defaults)
- `PatchDiscriminator` refactored: `self.model` (single `nn.Sequential`) split into named layers
  `layer1`–`layer4` + `final`; new `forward_with_features()` method returns logits + 4 intermediate
  feature maps
- Generator loss now: `L_G = L_GAN + L_L1 * lambda_l1 + Σ L1(fake_feat_i, real_feat_i.detach()) * lambda_fm`

---

## Why These Values

| Hyperparameter | Old | New | Rationale |
|---|---|---|---|
| `lambda_l1` | 100 | 10 | At 100, L1 dominated (~10× GAN loss). Reducing to 10 balances pixel fidelity with adversarial style pressure. |
| `lambda_fm` | — | 10 | Feature matching at each discriminator scale forces the generator to match real-glyph intermediate representations, not just final pixel values. 10 is a standard pix2pix-HD starting value. |

---

## ONNX Export Impact

No change to the ONNX input/output contract. The `FontGeneratorONNX` wrapper handles the
`style_glyph_0` extraction internally: `style_glyphs[:, 0]` is sliced from the `style_glyphs`
tensor that is already an ONNX input. Constant-folding during export will not collapse this path
(it depends on the dynamic `style_glyphs` input).

---

## Action Required

**Retrain from scratch.** The existing `models/v1/generator.onnx` (epoch_0200) was trained with
the buggy architecture and L1-dominant loss. It cannot be fine-tuned — the encoder weights were
trained on blank-canvas inputs and the skip connections were never style-conditioned.
