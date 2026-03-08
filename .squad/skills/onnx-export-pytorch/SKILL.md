# SKILL: PyTorch → ONNX Export Gotchas

**Author:** Major  
**Date:** 2026-03-08  
**Applies to:** Any PyTorch model exported to ONNX, especially GAN/UNet architectures

---

## Summary

Hard-won lessons from exporting the Cyrillic Font Generator cGAN to ONNX INT8.  
Covers two critical failure modes with known fixes.

---

## Gotcha 1 — `torch.compile` Checkpoint Key Prefix (`_orig_mod.`)

**Symptom:**
```
RuntimeError: Missing key(s): "layer.weight"
              Unexpected key(s): "_orig_mod.layer.weight"
```

**Root cause:** When `torch.compile` wraps a module, the compiled wrapper stores
the original module as `_orig_mod`. PyTorch's `state_dict()` serializes this
hierarchy, so all weight keys are prefixed with `_orig_mod.`.

When you later load the checkpoint into a plain (non-compiled) module, the keys
don't match.

**Fix:**

```python
def _strip_orig_mod(state_dict):
    """Strip '_orig_mod.' prefix added by torch.compile from state dict keys."""
    return {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()}

model.load_state_dict(_strip_orig_mod(ckpt["model_state"]))
```

This is safe and idempotent — non-compiled checkpoints pass through unchanged.

**When to apply:** Any inference/export script that loads checkpoints from training
runs where `torch.compile` may have been enabled.

---

## Gotcha 2 — `quantize_dynamic` + Stale `value_info` (ShapeInferenceError)

**Symptom:**
```
[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (512) vs (256)
```

**Root cause:** `quantize_dynamic` internally calls `replace_gemm_with_matmul()`,
which transposes Gemm weight initialisers in-place but does NOT update the
corresponding `value_info` shape annotations in the ONNX graph. When the modified
graph is saved and re-loaded with strict shape inference (`infer_shapes_path`),
the stale annotations conflict with the transposed tensor shapes.

**Fix:** Call `strip_initializer_value_info()` on the FP32 model BEFORE saving
the temp file passed to `quantize_dynamic`:

```python
def strip_initializer_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    init_names = {init.name for init in model.graph.initializer}
    stale = [vi for vi in model.graph.value_info if vi.name in init_names]
    for vi in stale:
        model.graph.value_info.remove(vi)
    return model

fp32_model = onnx.load(temp_fp32_path)
fp32_model = strip_initializer_value_info(fp32_model)  # ← MUST be before quantize_dynamic
onnx.save(fp32_model, cleaned_fp32_path)
quantize_dynamic(cleaned_fp32_path, int8_path, weight_type=QuantType.QUInt8)
```

The stripped entries are redundant — shapes are always recoverable from the
initialisers themselves — so removing them is safe.

---

## Gotcha 3 — ConvTranspose Cannot Be INT8 Quantized

**Symptom:** Model is larger than expected after INT8 quantization (e.g., 53 MB
instead of a hoped-for 23 MB).

**Root cause:** ONNX `IntegerOpsRegistry` has no `ConvTransposeInteger` operator.
`quantize_dynamic` simply skips all `ConvTranspose` nodes, leaving them in FP32.

**Impact on this project:** UNetGenerator has 7 decoder `ConvTranspose` layers
(~10.5M params, ~42 MB FP32). They cannot be INT8 quantized with onnxruntime
dynamic quantization.

**Workaround options:**
1. Accept the hybrid INT8+FP32 model (~53 MB, ~16 MB brotli) — production-ready.
2. Replace `ConvTranspose` with `Conv` + `Upsample` (bilinear) — allows full INT8
   but requires model redesign and retraining.
3. Custom static quantization pipeline — complex; not attempted.

---

## Recommended Export Pipeline Order

```
1. torch.onnx.export(model, dummy_inputs, fp32_path, ...)
2. onnx.checker.check_model(onnx.load(fp32_path))
3. fp32_model = strip_initializer_value_info(fp32_model)  ← Gotcha 2 fix
4. onnx.save(fp32_model, cleaned_fp32_path)
5. quantize_dynamic(cleaned_fp32_path, int8_path, weight_type=QuantType.QUInt8)
6. ort.InferenceSession(int8_path)  ← validate forward pass
```

And always strip `_orig_mod.` from checkpoint keys before step 0 (loading the model).

---

## Size Expectations (nf=32 UNetGenerator + StyleEncoder, ~21.6M params)

| Format | File size | Brotli estimate |
|--------|-----------|-----------------|
| FP32   | ~86 MB    | ~25 MB          |
| INT8   | ~53 MB    | ~17 MB          |
| FP16   | ~43 MB    | ~13 MB          |
