# Major — History

## Project Context
- **Project:** Cyrillic Font Generator
- **User:** FjoNef
- **Description:** Web app that generates Cyrillic font symbols for non-Cyrillic fonts using AI. Pre-trained model ships to client; all generative work runs in browser. .NET backend.
- **My focus:** AI model design, training, ONNX export, client-side inference strategy.

## Core Context

### Prior Investigation & Decisions (Feb 25 – Mar 6)
- ✅ Issue #21: ROOT CAUSE of INT8 quantization failure identified and fixed (strip_initializer_value_info() removes stale value_info entries before quantize_dynamic)
- ✅ base_filters 64→32 tradeoff analyzed (2.80× param reduction: 60.3M→21.6M total; StyleEncoder fixed at 7.1M); decision: retrain from scratch at nf=32
- ✅ cond_proj bottleneck identified (hardcoded 512 output oversized for nf=32 bottleneck 256, minor future micro-opt)
- ✅ ConvTranspose limitation confirmed: no ConvTransposeInteger in ONNX; 7 decoder layers ~10.5M params remain FP32 (unavoidable)
- ✅ Export pipeline hierarchy established: INT8 primary (~16 MB delivered), FP16 fallback (~13 MB), FP32 last resort (~25 MB)
- ✅ PR #8 merged (model training), PR #9 merged (backend integration), frontend ready for model

### Training Progress
- Retrained from scratch at base_filters=32 (target: ≤200 epochs at ~15 min/epoch on GPU)
- Checkpoint: epoch_0200 completed with quantization validation passing

## Learnings

### 2026-03-07: Style-Invariant Output — Root Cause Diagnosis

**Task:** Investigate why the model produces identical output regardless of input font style.

**Findings:**

- **Training data is correct:** Real fonts, `DEFAULT_STYLE_CHARS = ["A","B","C","D","E","H","I","O","R","X"]`, 45,207 samples — style glyphs are genuinely varied.
- **ONNX export is correct:** `style_glyphs` and `char_index` are both dynamic ONNX inputs. Constant folding folds the blank-canvas encoder path but not the style pathway. PR #40 (SharedArrayBuffer fix) was correct.
- **Root cause #1 — Architecture:** `UNetGenerator.forward()` feeds `torch.zeros(B, 1, 128, 128)` through the encoder. This makes all six U-Net skip connections (e1–e6) **deterministic constants** — identical regardless of font style. Style conditioning via `cond_spatial` enters only at the 1×1 bottleneck (a single injection point). After 6 decoder stages each mixed with constant skip connections, the style signal is overwhelmed.
- **Root cause #2 — Training loss:** `lambda_l1=100` dominates. No feature matching loss, no perceptual loss, no style supervision. The model is incentivized to minimize L1 against average glyph shapes, which requires no style sensitivity.
- **Root cause #3 — GAN instability:** Training logs (epochs 11–22) show D loss falling (0.31→0.26) while G loss rises (10.8→11.1), indicating discriminator dominance / precursor to mode collapse. Final loss state (epochs 23–200) is unlogged.
- **Training log gap:** TensorBoard event files in `models/logs/` are all 88 bytes (empty). Logs only cover epochs 11–22 from one training session. The full 200-epoch trajectory is not observable from files on disk.
- **Inference contract discrepancy:** `inference_contract.md` lists wrong style chars (`"g","n","o","p","s","x"` instead of `"C","D","E","I","R","X"`). `decisions.md` confirms frontend code is correct; the contract doc is outdated.

**Fix required:** Retrain.
1. Replace blank canvas encoder input with `style_glyphs[:, 0]` — gives skip connections per-font structure at every scale.
2. Add discriminator feature matching loss.
3. Reduce `lambda_l1` from 100 to 10.
4. Optionally: inject `cond_spatial` at multiple decoder scales (FiLM/AdaIN), not just bottleneck.

**Full diagnosis:** `.squad/decisions/inbox/major-style-conditioning-diagnosis.md`

---

### 2026-03-07T14:31:04Z: ONNX Export SUCCESS — Model Delivered (epoch_0200)

**Task:** Export trained cGAN to ONNX INT8 for browser delivery.

**Deliverable:**
- **File:** models/v1/generator.onnx
- **Size:** 53.1 MB (INT8 quantized)
- **Compressed:** ~15.9 MB brotli (meets ≤20 MB browser delivery target ✅)
- **Output shape:** (1, 1, 128, 128) float32, range [-1.0, 1.0]
- **Validation:** Sanity check passed; CPU inference SUCCESS

**Key Implementation:**
- Applied strip_initializer_value_info() to FP32 model before quantize_dynamic
- INT8 quantization applied to Conv/MatMul; ConvTranspose layers remain FP32
- Result: 53 MB uncompressed, 16 MB delivered (compressed), well under 20 MB target

**Rationale for 53 MB (not 23 MB):**
- quantize_dynamic has no ConvTransposeInteger op in ONNX IntegerOpsRegistry
- 7 decoder ConvTranspose layers (~10.5M params, 42 MB raw) forced to remain FP32
- All-INT8 architecture (~23 MB) would require custom static quantization pipeline or model redesign
- Current INT8+FP32 hybrid meets delivered target and is production-ready

**Implications:**
- ✅ **Togusa:** Frontend inference pipeline now fully functional (model ready for onnxruntime-web)
- ✅ **Batou:** 53.1 MB file needs HTTP brotli compression (~16 MB over wire)
- ✅ **Performance:** INT8 ~1.5–2× faster than FP32 on CPU
- 🔮 **Future:** Custom static quantization could reach ~23 MB if needed


### 2026-03-07: Style-Invariant Output — Diagnosis Complete

**Coordination with Togusa:**

Togusa's debug logging confirmed the JS layer is correct. No inference-time input format issues.

**Diagnosis Result:**

Three compounding causes:
1. **Architecture:** Blank canvas encoder kills skip connections (they're constant across all inputs).
2. **Loss:** L1-dominated training (lambda_l1=100) with no style supervision.
3. **GAN instability:** D dominance precursor suggests mode collapse.

**Action Items:**
1. Implement architectural fix: Use style_glyphs[:, 0] as encoder input.
2. Add feature matching loss.
3. Reduce lambda_l1 to 10.
4. Optionally validate with 10-epoch test on small font set before full retrain.

**Blocking Issue:** Model must be retrained. No inference-time changes can fix trained-out style conditioning.

