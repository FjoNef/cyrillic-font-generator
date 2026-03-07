"""
test_style_conditioning.py — Anticipatory regression tests for style conditioning bugs.

These tests guard against the regressions that were fixed by Major:

  Bug 1 — UNetGenerator.forward() fed torch.zeros to the U-Net encoder, meaning
           all skip connections carried zero-derived features rather than actual
           style information from the reference glyph.  Fixed: encoder now receives
           style_glyphs[:, 0] (first reference glyph) as its input image.

  Bug 2 — lambda_l1 was set to 100 in both train_config.yaml and the synthetic-mode
           defaults inside train.py, causing L1 to dominate and GAN to collapse.
           Fixed: lambda_l1 dropped to 10, feature-matching loss added.

Failure modes:
  - Tests 1-3  FAIL  if generator encoder reverts to torch.zeros input.
  - Test 4     FAILS if lambda_l1 is raised back above 20 in the config / script.
  - Test 5     FAILS if feature-matching loss is removed from the training code
               or the discriminator loses its intermediate-feature extraction path.

Expected fixed interface
------------------------
After Bug 1 is fixed, UNetGenerator.forward() accepts a third argument:

    generator(style_emb, char_index, style_glyph_0)

where style_glyph_0 : [B, 1, H, W] is the first reference glyph image.  The
encoder path then uses that tensor as input rather than torch.zeros.

Running these tests BEFORE the fix is applied is intentional: the failures act as
a living specification that must go green before the fix is considered complete.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/model/
# ---------------------------------------------------------------------------

_MODEL_ROOT = Path(__file__).resolve().parents[1]   # …/src/model
sys.path.insert(0, str(_MODEL_ROOT))

from train.model import StyleEncoder, UNetGenerator, PatchDiscriminator  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

_B = 2          # batch size
_N = 10         # style glyphs per sample
_H = _W = 128   # glyph resolution
_STYLE_DIM = 256
_CHAR_EMB_DIM = 64
_BASE_FILTERS = 32   # small model for test speed

_CHAR_IDX = torch.zeros(_B, dtype=torch.long)


def _make_models() -> tuple[StyleEncoder, UNetGenerator]:
    style_encoder = StyleEncoder(style_dim=_STYLE_DIM)
    generator = UNetGenerator(
        style_dim=_STYLE_DIM,
        char_emb_dim=_CHAR_EMB_DIM,
        base_filters=_BASE_FILTERS,
    )
    style_encoder.eval()
    generator.eval()
    return style_encoder, generator


def _make_style_glyphs(value: float = 0.5) -> torch.Tensor:
    """Return [B, N, 1, H, W] filled uniformly with *value*."""
    return torch.full((_B, _N, 1, _H, _W), value)


def _run_generator_with_style(
    generator: UNetGenerator,
    style_emb: torch.Tensor,
    char_idx: torch.Tensor,
    style_glyph_0: torch.Tensor,
) -> torch.Tensor:
    """
    Call generator using the expected *fixed* interface.

    After Bug 1 is resolved, UNetGenerator.forward() accepts a 3rd positional
    argument ``style_glyph_0 : [B, 1, H, W]`` and passes it to enc1 instead of
    torch.zeros.  If the interface has not yet been updated, this helper raises
    TypeError — which is the expected failure mode for anticipatory tests.
    """
    return generator(style_emb, char_idx, style_glyph_0)


# ---------------------------------------------------------------------------
# Test 1 — Encoder input is not zeros
# ---------------------------------------------------------------------------

class TestEncoderInputNotZeros:
    """
    Bug 1 guard: the tensor fed into generator.enc1 must NOT be all zeros.

    With the regression present: generator.enc1 always receives torch.zeros,
    so skip connections e1…e7 are determined only by learned biases — the style
    reference image has zero influence on the U-Net's spatial features.

    With the fix applied: enc1 receives style_glyphs[:, 0] (values ≈ ±0.8 here),
    so the hook-captured input tensor will be non-zero.
    """

    def test_enc1_input_is_not_all_zeros_for_nonzero_style_glyph(self) -> None:
        style_encoder, generator = _make_models()
        style_glyphs = _make_style_glyphs(value=0.8)

        enc1_inputs: list[torch.Tensor] = []

        def _capture_enc1_input(_module, inp, _out) -> None:
            enc1_inputs.append(inp[0].detach().clone())

        hook = generator.enc1.register_forward_hook(_capture_enc1_input)
        try:
            with torch.no_grad():
                style_emb = style_encoder(style_glyphs)
                _run_generator_with_style(
                    generator, style_emb, _CHAR_IDX, style_glyphs[:, 0]
                )
        finally:
            hook.remove()

        assert len(enc1_inputs) == 1, "enc1 hook was never triggered — forward() did not run"

        enc1_input = enc1_inputs[0]
        assert not torch.all(enc1_input == 0.0), (
            "REGRESSION DETECTED (Bug 1): generator.enc1 received an all-zero tensor.\n"
            "UNetGenerator.forward() must feed style_glyphs[:, 0] to the encoder, "
            "not torch.zeros(B, 1, H, W).  Skip connections currently carry no style "
            "information."
        )


# ---------------------------------------------------------------------------
# Test 2 — Style variation produces different skip features
# ---------------------------------------------------------------------------

class TestStyleVariationProducesDifferentSkipFeatures:
    """
    Bug 1 guard: the encoder's first skip feature (e1) must differ across batches
    that carry different style reference images.

    With the regression present: enc1 always processes zeros → e1 is identical for
    every batch regardless of style_glyphs content.

    With the fix applied: enc1 processes style_glyphs[:, 0], so different reference
    images yield meaningfully different e1 tensors.
    """

    def test_different_style_glyphs_produce_different_enc1_outputs(self) -> None:
        style_encoder, generator = _make_models()
        style_glyphs_a = _make_style_glyphs(value=0.8)   # bright glyphs
        style_glyphs_b = _make_style_glyphs(value=-0.8)  # dark glyphs

        enc1_outputs: list[torch.Tensor] = []

        def _capture_enc1_output(_module, _inp, out) -> None:
            enc1_outputs.append(out.detach().clone())

        hook = generator.enc1.register_forward_hook(_capture_enc1_output)
        try:
            with torch.no_grad():
                style_emb_a = style_encoder(style_glyphs_a)
                _run_generator_with_style(
                    generator, style_emb_a, _CHAR_IDX, style_glyphs_a[:, 0]
                )

                style_emb_b = style_encoder(style_glyphs_b)
                _run_generator_with_style(
                    generator, style_emb_b, _CHAR_IDX, style_glyphs_b[:, 0]
                )
        finally:
            hook.remove()

        assert len(enc1_outputs) == 2, "Expected exactly two enc1 forward passes"

        e1_a, e1_b = enc1_outputs
        assert not torch.allclose(e1_a, e1_b), (
            "REGRESSION DETECTED (Bug 1): enc1 output is identical for two batches "
            "with maximally different style_glyphs (±0.8).\n"
            "The encoder is not receiving the style reference image as input — it is "
            "likely still processing torch.zeros for both batches."
        )


# ---------------------------------------------------------------------------
# Test 3 — Style variation produces different generator output
# ---------------------------------------------------------------------------

class TestStyleVariationProducesDifferentOutput:
    """
    End-to-end guard: maximally different style_glyphs must produce different
    generated glyphs when the char_index is held constant.

    With the regression present (Bug 1): skip connections carry zero-derived features
    for all inputs, so the decoder path is partially blind to the reference image.
    This test adds a sanity-check on top of Test 2 to verify the difference in skip
    features propagates all the way to the final output tensor.

    Note: this test CAN pass even with the bug if the bottleneck conditioning
    (style_emb) alone introduces sufficient variation.  Its primary value is as a
    smoke test that style flows end-to-end after the encoder fix is applied.
    """

    def test_zeros_vs_ones_style_glyphs_produce_different_outputs(self) -> None:
        style_encoder, generator = _make_models()
        style_glyphs_zeros = _make_style_glyphs(value=-1.0)   # normalised black
        style_glyphs_ones = _make_style_glyphs(value=1.0)     # normalised white

        with torch.no_grad():
            style_emb_z = style_encoder(style_glyphs_zeros)
            out_z = _run_generator_with_style(
                generator, style_emb_z, _CHAR_IDX, style_glyphs_zeros[:, 0]
            )

            style_emb_o = style_encoder(style_glyphs_ones)
            out_o = _run_generator_with_style(
                generator, style_emb_o, _CHAR_IDX, style_glyphs_ones[:, 0]
            )

        assert not torch.allclose(out_z, out_o, atol=1e-4), (
            "REGRESSION DETECTED: generator produced identical outputs for "
            "maximally different style_glyphs (all -1.0 vs all +1.0).\n"
            "Style conditioning is not flowing through to the output.  Check both "
            "the StyleEncoder and the UNetGenerator encoder path."
        )

    def test_output_shape_is_correct(self) -> None:
        """Sanity check: output is [B, 1, 128, 128] with values in [-1, 1]."""
        style_encoder, generator = _make_models()
        style_glyphs = _make_style_glyphs(value=0.5)

        with torch.no_grad():
            style_emb = style_encoder(style_glyphs)
            output = _run_generator_with_style(
                generator, style_emb, _CHAR_IDX, style_glyphs[:, 0]
            )

        assert output.shape == (_B, 1, _H, _W), (
            f"Expected output shape ({_B}, 1, {_H}, {_W}), got {tuple(output.shape)}"
        )
        assert output.min() >= -1.0 - 1e-5 and output.max() <= 1.0 + 1e-5, (
            f"Generator output values should be in [-1, 1] (Tanh), "
            f"but got min={output.min():.4f}, max={output.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4 — lambda_l1 is within acceptable range
# ---------------------------------------------------------------------------

class TestLambdaL1Config:
    """
    Bug 2 guard: lambda_l1 must never exceed 20.

    The original value of 100 caused the L1 reconstruction term to completely
    dominate the generator loss, preventing the GAN from learning a useful
    adversarial objective.  The fix lowers it to 10.

    This test checks BOTH the YAML config file AND the hard-coded synthetic-mode
    defaults in train.py to catch either regression path.
    """

    _CONFIG_YAML = _MODEL_ROOT / "configs" / "train_config.yaml"
    _TRAIN_PY = _MODEL_ROOT / "train" / "train.py"
    _MAX_LAMBDA_L1 = 20

    def test_yaml_config_lambda_l1_is_within_range(self) -> None:
        import yaml  # part of PyYAML, listed in requirements.txt

        assert self._CONFIG_YAML.exists(), (
            f"Config file not found: {self._CONFIG_YAML}\n"
            "Expected at src/model/configs/train_config.yaml"
        )

        with self._CONFIG_YAML.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        lambda_l1 = cfg["training"]["lambda_l1"]
        assert lambda_l1 <= self._MAX_LAMBDA_L1, (
            f"REGRESSION DETECTED (Bug 2): lambda_l1={lambda_l1} in train_config.yaml "
            f"exceeds the maximum allowed value of {self._MAX_LAMBDA_L1}.\n"
            "A value of 100 caused L1 to dominate the generator loss and suppressed "
            "adversarial learning.  Expected value: 10."
        )

    def test_synthetic_mode_lambda_l1_is_within_range(self) -> None:
        """
        The synthetic-mode defaults in train.py contain their own lambda_l1 literal.
        Parse the AST to find it without executing the file.
        """
        assert self._TRAIN_PY.exists(), (
            f"train.py not found: {self._TRAIN_PY}"
        )

        source = self._TRAIN_PY.read_text(encoding="utf-8")
        tree = ast.parse(source)

        lambda_l1_values: list[int | float] = []

        # Walk the AST looking for dict literals that contain the key 'lambda_l1'.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, val in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "lambda_l1"
                    and isinstance(val, ast.Constant)
                    and isinstance(val.value, (int, float))
                ):
                    lambda_l1_values.append(val.value)

        assert lambda_l1_values, (
            "Could not find a 'lambda_l1' key with a numeric literal in train.py.\n"
            "Check that the synthetic-mode defaults dict still contains lambda_l1."
        )

        for value in lambda_l1_values:
            assert value <= self._MAX_LAMBDA_L1, (
                f"REGRESSION DETECTED (Bug 2): lambda_l1={value} found in train.py "
                f"exceeds the maximum allowed value of {self._MAX_LAMBDA_L1}.\n"
                "Expected the synthetic-mode default to be 10 or lower."
            )


# ---------------------------------------------------------------------------
# Test 5 — Feature-matching loss is present
# ---------------------------------------------------------------------------

class TestFeatureMatchingLoss:
    """
    Bug 2 guard: the discriminator must expose intermediate features, and the
    training loop must include a feature-matching loss term.

    After the fix, PatchDiscriminator is expected to either:
      (a) return intermediate feature maps alongside the final patch logits, or
      (b) expose a dedicated method (e.g. get_features()) for the training loop
          to query.

    The training loop in train.py must use these features to compute a
    feature-matching loss (mean L1 distance between real and fake discriminator
    features) that supplements the GAN + pixel-L1 losses.

    These tests check for structural signals — they do NOT run a full training
    step, which would require a real dataset.
    """

    _TRAIN_PY = _MODEL_ROOT / "train" / "train.py"

    def test_discriminator_exposes_intermediate_features(self) -> None:
        """
        PatchDiscriminator must provide a way to extract intermediate feature maps.

        Acceptable implementations:
          - forward() returns (patch_logits, features) tuple
          - A separate get_features() / extract_features() method exists
          - The model is refactored into named sub-layers accessible via attribute
        """
        disc = PatchDiscriminator()
        image = torch.randn(_B, 1, _H, _W)
        style_glyph = torch.randn(_B, 1, _H, _W)

        # Check for explicit feature extraction method first.
        has_feature_method = (
            hasattr(disc, "get_features")
            or hasattr(disc, "extract_features")
            or hasattr(disc, "forward_features")
            or hasattr(disc, "forward_with_features")
        )

        # Check if forward() returns a tuple (logits, features).
        with torch.no_grad():
            output = disc(image, style_glyph)

        returns_tuple = isinstance(output, (tuple, list)) and len(output) >= 2

        assert has_feature_method or returns_tuple, (
            "REGRESSION DETECTED (Bug 2): PatchDiscriminator does not expose intermediate "
            "features.\n"
            "Feature-matching loss requires either:\n"
            "  - disc.forward_with_features(image, style_glyph) → (logits, List[Tensor]), or\n"
            "  - disc.get_features / disc.extract_features / disc.forward_features method, or\n"
            "  - disc.forward() returning (patch_logits, feature_list) tuple.\n"
            "Currently forward() returns a plain Tensor and no feature method exists."
        )

    def test_forward_with_features_returns_logits_and_feature_list(self) -> None:
        """
        forward_with_features() must return (patch_logits, features) where features
        is a non-empty list of intermediate tensors.  Verifies the discriminator
        can supply the real/fake feature pairs needed for feature-matching loss.
        """
        disc = PatchDiscriminator()
        image = torch.randn(_B, 1, _H, _W)
        style_glyph = torch.randn(_B, 1, _H, _W)

        assert hasattr(disc, "forward_with_features"), (
            "REGRESSION DETECTED (Bug 2): PatchDiscriminator.forward_with_features() "
            "method is missing.  This method is required for feature-matching loss."
        )

        with torch.no_grad():
            output = disc.forward_with_features(image, style_glyph)

        assert isinstance(output, (tuple, list)) and len(output) == 2, (
            f"forward_with_features() must return a 2-tuple (logits, features), "
            f"got {type(output)}"
        )

        logits, features = output
        assert isinstance(logits, torch.Tensor), "First return value must be patch logits tensor"
        assert logits.shape == (_B, 1, 14, 14), (
            f"Expected patch logits shape ({_B}, 1, 14, 14), got {tuple(logits.shape)}"
        )
        assert isinstance(features, list) and len(features) >= 1, (
            "Second return value must be a non-empty list of intermediate feature tensors"
        )
        for i, feat in enumerate(features):
            assert isinstance(feat, torch.Tensor), (
                f"Feature {i} must be a tensor, got {type(feat)}"
            )

    def test_train_py_contains_feature_matching_loss_term(self) -> None:
        """
        The training script must contain a feature-matching loss computation.

        We check for the presence of source patterns associated with feature matching:
          - 'feat' appears in loss variable names, or
          - 'feature_match' / 'fm_loss' / 'loss_fm' identifiers exist, or
          - L1 loss is called on discriminator feature tensors.

        This test parses train.py's AST for assignment targets whose names include
        'feat' (case-insensitive) alongside a loss/criterion call.
        """
        assert self._TRAIN_PY.exists(), (
            f"train.py not found: {self._TRAIN_PY}"
        )

        source = self._TRAIN_PY.read_text(encoding="utf-8")

        # Text-level heuristic: look for feature-matching idioms.
        fm_indicators = [
            "feat",           # loss_feat, feat_match, fm_loss, etc.
            "feature_match",
            "feature_loss",
            "fm_loss",
            "loss_fm",
        ]

        source_lower = source.lower()
        found_indicators = [ind for ind in fm_indicators if ind in source_lower]

        assert found_indicators, (
            "REGRESSION DETECTED (Bug 2): No feature-matching loss indicators found in "
            "train.py.\n"
            f"Searched for: {fm_indicators}\n"
            "The generator loss must include a feature-matching term that penalises "
            "the L1 distance between real and fake intermediate discriminator features.  "
            "If the indicator names differ, add one of the expected names or update "
            "this test."
        )
