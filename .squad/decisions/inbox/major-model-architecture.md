# Model Architecture: cGAN with Style Encoder

**Date:** 2026-02-25  
**Author:** Major (AI/ML Engineer)  
**Status:** Implemented  
**PR:** #8

## Decision

Use a **conditional GAN (pix2pix-style)** architecture with:
1. **StyleEncoder** — Encodes N Latin reference glyphs into a fixed 256-dim style vector
2. **UNetGenerator** — Generates 128×128 Cyrillic glyph conditioned on character index + style
3. **PatchDiscriminator** — 70×70 PatchGAN for adversarial training

## Rationale

### StyleEncoder Design
- **Shared-weight CNN + mean-pooling** over N reference glyphs
- **Why mean-pooling:** Makes style representation permutation-invariant (order doesn't matter)
- **Why shared weights:** Each glyph is encoded independently, then aggregated — flexible and efficient
- **Alternative considered:** Concatenate all glyphs → single CNN. Rejected: not scalable to variable N.

### UNetGenerator Design
- **U-Net with skip connections:** Standard for image-to-image tasks, preserves spatial structure
- **Blank canvas input (all zeros):** Model learns to generate from scratch, no skeleton template needed
- **Conditioning at bottleneck:** Style vector + character embedding concatenated at 4×4 spatial resolution
- **Why concatenation, not AdaIN:** Simpler ONNX export, no dynamic BatchNorm issues
- **Character embedding:** 128-dim learned embedding expanded to spatial 4×4 grid
- **Alternative considered:** Inject conditioning at each layer. Rejected: harder to export, minimal quality gain.

### PatchDiscriminator Design
- **70×70 PatchGAN:** Standard pix2pix discriminator, penalizes per-patch realism
- **Conditioning:** Concatenate generated glyph + one style reference glyph (first of 10)
- **Why PatchGAN:** Better than global discriminator for local texture quality (stroke consistency)
- **Alternative considered:** Global discriminator. Rejected: tends to average out fine details.

### Training Loss
- **Adversarial loss (BCE):** Standard GAN objective — fool the discriminator
- **L1 reconstruction loss:** Pixel-wise distance to real glyph (weighted 100:1 with adversarial)
- **Why L1, not L2:** Less blurry outputs, better for sharp font strokes
- **Lambda = 100:** Standard pix2pix weight, prioritizes reconstruction over pure adversarial realism

## Alternatives Considered

1. **VAE (Variational Autoencoder):**
   - **Pro:** Stable training, no mode collapse
   - **Con:** Blurry outputs, less suitable for sharp font glyphs
   - **Verdict:** GANs produce sharper results for this use case

2. **Diffusion Models:**
   - **Pro:** State-of-art quality
   - **Con:** Too slow for browser inference (50+ steps), model size too large
   - **Verdict:** Deferred to future iteration

3. **Direct CNN (no GAN):**
   - **Pro:** Simpler, no adversarial training instability
   - **Con:** Blurry outputs without adversarial signal
   - **Verdict:** L1-only training produces inferior results

## Model Size

- **Generator parameters:** ~15-20M (depends on exact U-Net depth)
- **ONNX (float32):** ~40-50 MB
- **ONNX (INT8 quantized):** ~15-20 MB ✅ Browser delivery target

## Implementation

Files created in PR #8:
- `models/train/model.py` — Architecture definitions
- `models/train/train.py` — Training loop
- `models/train/export.py` — ONNX export with quantization

## Impact

- **Training:** Can train on Google Fonts corpus (100-400 fonts)
- **Inference:** Model fits browser delivery constraints (<20 MB)
- **Quality:** GAN training should produce sharp, stylistically consistent glyphs
- **Frontend:** Zero changes needed — contract already defined and implemented

## Next Steps

1. Acquire training data (Google Fonts with Latin+Cyrillic)
2. Train for 200 epochs (~4-8 hours on GPU)
3. Export to `models/v1/generator.onnx`
4. Validate quality in browser

## Related Decisions

- [Model tensor contract](../decisions.md#2026-02-25-ml-engineering-decisions) — Locked input/output shapes
- [Inference pipeline](../decisions.md#2026-02-25t143900-inference-pipeline-implementation) — Frontend integration