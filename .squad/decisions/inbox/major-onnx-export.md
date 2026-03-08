# Decision: ONNX Export — torch.compile State Dict Key Handling

**Date:** 2026-03-08  
**Author:** Major (AI/ML Engineer)  
**Status:** Resolved — fix committed to `dev`

---

## Context

After the full model retrain (epoch_0200, 2026-03-08), the ONNX export script
`src/model/export/export_onnx.py` crashed with a `RuntimeError` when loading the
checkpoint:

```
Missing key(s): "char_embedding.weight", ...
Unexpected key(s): "_orig_mod.char_embedding.weight", ...
```

**Root cause:** Training was run with `use_compile: true` in `train_config.yaml`.
When `torch.compile` wraps a module, it stores weights under an inner `_orig_mod`
attribute. PyTorch's checkpoint serialization preserves this prefix in all keys.
The export script instantiated a plain `UNetGenerator` (no compile) and called
`load_state_dict()` directly — key names did not match.

---

## Decision

**Fix:** Added `_strip_orig_mod(state_dict)` helper to `export_onnx.py` that strips
the `_orig_mod.` prefix from all checkpoint keys before loading. Applied to both
`style_encoder_state` and `generator_state` dicts.

```python
def _strip_orig_mod(state_dict):
    return {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}
```

This is transparent to the caller and handles both compiled and non-compiled checkpoints.

---

## Implications

- **Any future export** from a `use_compile: true` training run will work without
  manual intervention.
- The fix is backward-compatible: non-compiled checkpoints have no `_orig_mod.` prefix
  so the helper is a no-op.
- The `_strip_orig_mod()` pattern should be applied in any other script that loads
  checkpoints (e.g. evaluation, fine-tuning scripts).

---

## Export Result (Post-Fix)

| Metric | Value |
|---|---|
| Checkpoint | `models/checkpoints/epoch_0200.pth` |
| Format | INT8 dynamic quantization |
| Output | `models/v1/generator.onnx` |
| Size | 53.1 MB |
| Brotli estimate | ~15.9 MB ✅ |
| Output shape | (B, 1, 128, 128) float32 |
| Value range | [-1.0, 1.0] ✅ |
| Commit | `ceab05d` on `dev` |
