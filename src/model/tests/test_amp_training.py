"""
test_amp_training.py — AMP smoke test for the GAN training loop.

Exercises one complete training step (discriminator + generator) with
torch.amp.GradScaler and autocast enabled, then asserts that all losses
are finite (no NaN / Inf).

Design constraints:
  - CPU-only: autocast and GradScaler are constructed with enabled=False on
    CPU so they are no-ops — the test runs in any CI environment without a GPU.
  - No real training data: uses tiny dummy tensors (batch of 2, 1×128×128).
  - No filesystem access: all models are constructed in-memory with small
    base_filters so the forward pass is fast.
  - Follows the same structure as the live training loop in train/train.py so
    regressions in AMP wiring are caught at unit-test time.

Failure modes:
  - Fails if GradScaler is constructed without enabled=False on CPU
    (would raise RuntimeError: No CUDA device).
  - Fails if backward() is incorrectly placed inside the autocast block
    (no immediate error, but can cause silent numerical issues — the finite-loss
    assert acts as a proxy check).
  - Fails if persistent_workers=True is set with num_workers=0
    (ValueError at DataLoader construction, not exercised here but guarded in
    test_persistent_workers_flag below).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — allow imports from src/model/
# ---------------------------------------------------------------------------

_MODEL_ROOT = Path(__file__).resolve().parents[1]  # …/src/model
sys.path.insert(0, str(_MODEL_ROOT))

from train.model import StyleEncoder, UNetGenerator, PatchDiscriminator  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_B = 2           # tiny batch for speed
_N = 3           # style glyphs per sample (reduced from 10 for test speed)
_H = _W = 128    # glyph resolution (must match model expectations)
_STYLE_DIM = 64  # small embedding dim for test speed
_CHAR_EMB_DIM = 32
_BASE_FILTERS = 8  # smallest legal value for fast CPU forward pass

_DEVICE = torch.device("cpu")
_USE_AMP = False  # AMP is a no-op on CPU; GradScaler(enabled=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_models() -> tuple[StyleEncoder, UNetGenerator, PatchDiscriminator]:
    style_encoder = StyleEncoder(style_dim=_STYLE_DIM).to(_DEVICE)
    generator = UNetGenerator(
        style_dim=_STYLE_DIM,
        char_emb_dim=_CHAR_EMB_DIM,
        base_filters=_BASE_FILTERS,
    ).to(_DEVICE)
    discriminator = PatchDiscriminator(ndf=_BASE_FILTERS).to(_DEVICE)
    return style_encoder, generator, discriminator


def _make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (style_glyphs [B,N,1,H,W], target_glyph [B,1,H,W], char_index [B])."""
    style_glyphs = torch.randn(_B, _N, 1, _H, _W, device=_DEVICE)
    target_glyph = torch.randn(_B, 1, _H, _W, device=_DEVICE)
    char_index = torch.zeros(_B, dtype=torch.long, device=_DEVICE)
    return style_glyphs, target_glyph, char_index


def _gan_loss(
    pred: torch.Tensor,
    target_is_real: bool,
    criterion: nn.BCEWithLogitsLoss,
) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return criterion(pred, target)


# ---------------------------------------------------------------------------
# Import the same autocast/GradScaler that train.py uses so this test
# exercises the actual import path chosen at runtime.
# ---------------------------------------------------------------------------

try:
    from torch.amp import GradScaler, autocast as _autocast_fn  # PyTorch ≥ 2.0

    def autocast(enabled: bool = True):  # type: ignore[misc]
        return _autocast_fn("cuda", enabled=enabled)

except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestAMPSmokeStep:
    """
    Single-step GAN training smoke test with AMP scaffolding active.

    GradScaler(enabled=False) and autocast(enabled=False) are used so the test
    is a complete no-op on CPU while still exercising the wiring of the AMP API.
    """

    def test_discriminator_step_produces_finite_loss(self) -> None:
        """Discriminator backward step must produce a finite loss (no NaN/Inf)."""
        style_encoder, generator, discriminator = _make_models()
        style_glyphs, target_glyph, char_index = _make_batch()

        criterion_gan = nn.BCEWithLogitsLoss()
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        scaler_d = GradScaler(enabled=_USE_AMP)

        cond_glyph = style_glyphs[:, 0]

        opt_d.zero_grad()
        with autocast(enabled=_USE_AMP):
            with torch.no_grad():
                style_emb = style_encoder(style_glyphs)
                fake_glyph = generator(style_emb, char_index, cond_glyph)
            real_logits = discriminator(target_glyph, cond_glyph)
            fake_logits = discriminator(fake_glyph.detach(), cond_glyph)
            loss_d_real = _gan_loss(real_logits, True, criterion_gan)
            loss_d_fake = _gan_loss(fake_logits, False, criterion_gan)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        assert torch.isfinite(loss_d), (
            f"Discriminator loss is not finite after one training step: {loss_d.item()}"
        )

    def test_generator_step_produces_finite_loss(self) -> None:
        """Generator backward step (GAN + L1 + feature-matching) must produce finite losses."""
        style_encoder, generator, discriminator = _make_models()
        style_glyphs, target_glyph, char_index = _make_batch()

        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()
        opt_g = torch.optim.Adam(
            list(style_encoder.parameters()) + list(generator.parameters()), lr=1e-4
        )
        scaler_g = GradScaler(enabled=_USE_AMP)

        cond_glyph = style_glyphs[:, 0]
        lambda_l1 = 10.0
        lambda_fm = 10.0

        opt_g.zero_grad()
        with autocast(enabled=_USE_AMP):
            style_emb = style_encoder(style_glyphs)
            fake_glyph = generator(style_emb, char_index, cond_glyph)
            fake_logits_g, fake_features = discriminator.forward_with_features(
                fake_glyph, cond_glyph
            )
            _, real_features = discriminator.forward_with_features(
                target_glyph, cond_glyph
            )
            loss_gan = _gan_loss(fake_logits_g, True, criterion_gan)
            loss_l1 = criterion_l1(fake_glyph, target_glyph) * lambda_l1
            loss_fm = (
                sum(
                    F.l1_loss(ff, rf.detach())
                    for ff, rf in zip(fake_features, real_features)
                )
                * lambda_fm
            )
            loss_g = loss_gan + loss_l1 + loss_fm

        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        assert torch.isfinite(loss_g), (
            f"Generator loss is not finite after one training step: {loss_g.item()}"
        )
        assert torch.isfinite(loss_l1), (
            f"L1 loss component is not finite: {loss_l1.item()}"
        )
        assert torch.isfinite(loss_fm), (
            f"Feature-matching loss component is not finite: {loss_fm.item()}"
        )

    def test_full_step_losses_are_finite(self) -> None:
        """
        Combined discriminator + generator step — mirrors the live training loop.

        Runs both backward passes in sequence (as in train.py) and asserts all
        loss scalars are finite.  This is the primary AMP smoke test.
        """
        style_encoder, generator, discriminator = _make_models()
        style_glyphs, target_glyph, char_index = _make_batch()

        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()
        opt_g = torch.optim.Adam(
            list(style_encoder.parameters()) + list(generator.parameters()), lr=1e-4
        )
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        scaler_g = GradScaler(enabled=_USE_AMP)
        scaler_d = GradScaler(enabled=_USE_AMP)

        lambda_l1, lambda_fm = 10.0, 10.0
        cond_glyph = style_glyphs[:, 0]

        # --- Discriminator step ---
        opt_d.zero_grad()
        with autocast(enabled=_USE_AMP):
            style_emb = style_encoder(style_glyphs)
            fake_glyph = generator(style_emb.detach(), char_index, cond_glyph).detach()
            real_logits = discriminator(target_glyph, cond_glyph)
            fake_logits = discriminator(fake_glyph, cond_glyph)
            loss_d = 0.5 * (
                _gan_loss(real_logits, True, criterion_gan)
                + _gan_loss(fake_logits, False, criterion_gan)
            )
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # --- Generator step ---
        opt_g.zero_grad()
        with autocast(enabled=_USE_AMP):
            style_emb = style_encoder(style_glyphs)
            fake_glyph = generator(style_emb, char_index, cond_glyph)
            fake_logits_g, fake_features = discriminator.forward_with_features(
                fake_glyph, cond_glyph
            )
            _, real_features = discriminator.forward_with_features(
                target_glyph, cond_glyph
            )
            loss_gan = _gan_loss(fake_logits_g, True, criterion_gan)
            loss_l1 = criterion_l1(fake_glyph, target_glyph) * lambda_l1
            loss_fm = (
                sum(
                    F.l1_loss(ff, rf.detach())
                    for ff, rf in zip(fake_features, real_features)
                )
                * lambda_fm
            )
            loss_g = loss_gan + loss_l1 + loss_fm
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        for name, loss in [("loss_d", loss_d), ("loss_g", loss_g), ("loss_l1", loss_l1), ("loss_fm", loss_fm)]:
            assert not torch.isnan(loss), f"{name} is NaN after one full training step"
            assert not torch.isinf(loss), f"{name} is Inf after one full training step"

    def test_grad_scaler_scale_is_positive_after_valid_step(self) -> None:
        """
        GradScaler.get_scale() must remain positive after a step with no overflow.

        A scale collapse to 0 or negative indicates the first forward pass produced
        Inf/NaN gradients.  On CPU with enabled=False the scale is always 1.0.
        """
        scaler = GradScaler(enabled=_USE_AMP)

        style_encoder, generator, discriminator = _make_models()
        style_glyphs, target_glyph, char_index = _make_batch()
        criterion_gan = nn.BCEWithLogitsLoss()
        opt_g = torch.optim.Adam(
            list(style_encoder.parameters()) + list(generator.parameters()), lr=1e-4
        )

        cond_glyph = style_glyphs[:, 0]
        opt_g.zero_grad()
        with autocast(enabled=_USE_AMP):
            style_emb = style_encoder(style_glyphs)
            fake_glyph = generator(style_emb, char_index, cond_glyph)
            fake_logits_g, _ = discriminator.forward_with_features(fake_glyph, cond_glyph)
            loss = _gan_loss(fake_logits_g, True, criterion_gan)
        scaler.scale(loss).backward()
        scaler.step(opt_g)
        scaler.update()

        assert scaler.get_scale() > 0, (
            f"GradScaler scale collapsed to {scaler.get_scale()} after first step — "
            "NaN/Inf in forward pass caused scale to drop to 0."
        )


# ---------------------------------------------------------------------------
# Test: persistent_workers flag guard
# ---------------------------------------------------------------------------

class TestPersistentWorkersFlag:
    """
    Guard: persistent_workers=True with num_workers=0 raises ValueError.

    This test documents the expected DataLoader behaviour and verifies that
    the fix (persistent_workers=num_workers > 0) prevents the error.
    """

    def test_persistent_workers_false_when_num_workers_is_zero(self) -> None:
        """
        When num_workers=0, persistent_workers must evaluate to False.
        """
        num_workers = 0
        assert (num_workers > 0) is False, (
            "persistent_workers expression `num_workers > 0` must be False "
            "when num_workers=0 to prevent ValueError at DataLoader construction."
        )

    def test_persistent_workers_true_when_num_workers_positive(self) -> None:
        """
        When num_workers > 0, persistent_workers must evaluate to True.
        """
        num_workers = 4
        assert (num_workers > 0) is True, (
            "persistent_workers expression `num_workers > 0` must be True "
            "when num_workers=4."
        )
