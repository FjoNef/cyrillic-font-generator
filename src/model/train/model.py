"""
model.py — PyTorch model definitions for the Cyrillic font generator.

Architecture overview
---------------------

  StyleEncoder
    Input:  style_glyphs  [B, N, 1, H, W]   — N Latin reference glyphs
    Output: style_emb     [B, 256]           — global style embedding

  UNetGenerator
    Input:  style_emb     [B, 256]
            char_index    [B]  (int64, used to look up char_emb [B, 64])
    Output: glyph         [B, 1, 128, 128]   — generated glyph in [-1, 1]

  PatchDiscriminator (PatchGAN 70×70)
    Input:  real_or_fake  [B, 1, 128, 128]
            style_glyph   [B, 1, 128, 128]   — conditioning image (one ref glyph)
    Output: patch_logits  [B, 1, 14, 14]     — real/fake score per patch

Total parameter budget: StyleEncoder ~1M, UNetGenerator ~8M, Discriminator ~1M.
Only StyleEncoder + UNetGenerator are exported to ONNX.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 4,
    stride: int = 2,
    padding: int = 1,
    norm: bool = True,
    activation: str = "leaky",
) -> nn.Sequential:
    """Convolution → optional InstanceNorm → activation."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _deconv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 4,
    stride: int = 2,
    padding: int = 1,
    dropout: bool = False,
) -> nn.Sequential:
    """Transposed-convolution → InstanceNorm → optional Dropout → ReLU."""
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.InstanceNorm2d(out_ch),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# StyleEncoder
# ---------------------------------------------------------------------------

class StyleEncoder(nn.Module):
    """
    Encodes N Latin reference glyphs into a single style embedding vector.

    The encoder processes each glyph independently with a shared CNN, then
    aggregates across the N glyphs using mean-pooling to produce a
    permutation-invariant style representation.

    Input:  style_glyphs  [B, N, 1, 128, 128]  float32  values in [-1, 1]
    Output: style_emb     [B, 256]              float32
    """

    def __init__(self, in_channels: int = 1, style_dim: int = 256) -> None:
        super().__init__()
        # Shared CNN backbone — output spatial size for 128×128 input: 4×4
        self.encoder = nn.Sequential(
            _conv_block(in_channels, 64,  norm=False),   # [B, 64,  64, 64]
            _conv_block(64,  128),                        # [B, 128, 32, 32]
            _conv_block(128, 256),                        # [B, 256, 16, 16]
            _conv_block(256, 512),                        # [B, 512,  8,  8]
            _conv_block(512, 512),                        # [B, 512,  4,  4]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)               # [B, 512, 1, 1]
        self.fc = nn.Linear(512, style_dim)               # [B, 256]

    def forward(self, style_glyphs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        style_glyphs : [B, N, 1, H, W]

        Returns
        -------
        style_emb : [B, style_dim]
        """
        B, N, C, H, W = style_glyphs.shape
        # Flatten batch and N dims, run shared encoder.
        x = style_glyphs.view(B * N, C, H, W)          # [B*N, 1, H, W]
        x = self.encoder(x)                             # [B*N, 512, 4, 4]
        x = self.pool(x).view(B * N, -1)               # [B*N, 512]
        x = self.fc(x)                                  # [B*N, style_dim]
        # Mean-pool over the N reference glyphs.
        x = x.view(B, N, -1).mean(dim=1)               # [B, style_dim]
        return x


# ---------------------------------------------------------------------------
# UNetGenerator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """
    U-Net generator conditioned on style embedding and target character index.

    The style embedding and character embedding are concatenated, projected to
    a spatial feature map, and injected at the U-Net bottleneck via
    concatenation before the first decoder block.

    Input:
        style_emb  : [B, style_dim]   float32
        char_index : [B]              int64   (values 0–65)
    Output:
        glyph      : [B, 1, 128, 128] float32  values in [-1, 1]

    Encoder path (downsampling):
        128 → 64 → 32 → 16 → 8 → 4 → 2 → 1  (6 stride-2 conv layers)
    Decoder path (upsampling with skip connections):
        1 → 2 → 4 → 8 → 16 → 32 → 64 → 128
    """

    NUM_CHARS = 66   # 33 uppercase + 33 lowercase Russian Cyrillic

    def __init__(
        self,
        style_dim: int = 256,
        char_emb_dim: int = 64,
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        nf = base_filters
        self.char_embedding = nn.Embedding(self.NUM_CHARS, char_emb_dim)

        # Project (style_dim + char_emb_dim) → 512 spatial bottleneck.
        # We produce a [B, 512, 1, 1] feature and tile it.
        self.cond_proj = nn.Sequential(
            nn.Linear(style_dim + char_emb_dim, 512),
            nn.ReLU(inplace=True),
        )

        # --- Encoder (image input path) ---
        # Input: blank 128×128 canvas (zeros) + spatially-tiled conditioning.
        # We inject conditioning at the very first conv so the network learns
        # to generate from scratch rather than transform an existing image.
        # Encoder expects 1 input channel (grayscale skeleton / blank canvas).
        self.enc1 = _conv_block(1,      nf,       norm=False)  # 64  → [nf,   64, 64]
        self.enc2 = _conv_block(nf,     nf * 2)               # 64  → [nf*2, 32, 32]
        self.enc3 = _conv_block(nf * 2, nf * 4)               # 32  → [nf*4, 16, 16]
        self.enc4 = _conv_block(nf * 4, nf * 8)               # 16  → [nf*8,  8,  8]
        self.enc5 = _conv_block(nf * 8, nf * 8)               #  8  → [nf*8,  4,  4]
        self.enc6 = _conv_block(nf * 8, nf * 8)               #  4  → [nf*8,  2,  2]
        self.enc7 = _conv_block(nf * 8, nf * 8, norm=False)   #  2  → [nf*8,  1,  1] (bottleneck)

        # --- Decoder ---
        # Bottleneck receives enc7 + projected conditioning (both [B, 512, 1, 1]).
        self.dec1 = _deconv_block(nf * 8 + 512, nf * 8, dropout=True)   # → [nf*8, 2, 2]
        self.dec2 = _deconv_block(nf * 8 * 2,   nf * 8, dropout=True)   # → [nf*8, 4, 4]  (+skip enc6)
        self.dec3 = _deconv_block(nf * 8 * 2,   nf * 8, dropout=True)   # → [nf*8, 8, 8]  (+skip enc5)
        self.dec4 = _deconv_block(nf * 8 * 2,   nf * 8)                  # → [nf*8,16,16]  (+skip enc4)
        self.dec5 = _deconv_block(nf * 8 * 2,   nf * 4)                  # → [nf*4,32,32]  (+skip enc3)
        self.dec6 = _deconv_block(nf * 4 * 2,   nf * 2)                  # → [nf*2,64,64]  (+skip enc2)
        self.dec7 = _deconv_block(nf * 2 * 2,   nf)                      # → [nf, 128,128] (+skip enc1)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, 1, kernel_size=4, stride=2, padding=1),  # shouldn't upsample here
            nn.Tanh(),
        )
        # Correct final layer: no upsampling needed after dec7 (already 128×128).
        self.final = nn.Sequential(
            nn.Conv2d(nf * 2, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        style_emb: torch.Tensor,   # [B, style_dim]
        char_index: torch.Tensor,  # [B] int64
    ) -> torch.Tensor:
        """
        Returns
        -------
        glyph : [B, 1, 128, 128]  float32  values in [-1, 1]
        """
        B = style_emb.shape[0]
        device = style_emb.device

        # Build conditioning vector.
        char_emb = self.char_embedding(char_index)              # [B, char_emb_dim]
        cond = torch.cat([style_emb, char_emb], dim=1)         # [B, style_dim+char_emb_dim]
        cond_feat = self.cond_proj(cond)                        # [B, 512]
        cond_spatial = cond_feat.view(B, 512, 1, 1)            # [B, 512, 1, 1]

        # Encoder: blank canvas input.
        x = torch.zeros(B, 1, 128, 128, device=device)
        e1 = self.enc1(x)    # [B, nf,   64, 64]
        e2 = self.enc2(e1)   # [B, nf*2, 32, 32]
        e3 = self.enc3(e2)   # [B, nf*4, 16, 16]
        e4 = self.enc4(e3)   # [B, nf*8,  8,  8]
        e5 = self.enc5(e4)   # [B, nf*8,  4,  4]
        e6 = self.enc6(e5)   # [B, nf*8,  2,  2]
        e7 = self.enc7(e6)   # [B, nf*8,  1,  1]  bottleneck

        # Inject conditioning at bottleneck.
        d = torch.cat([e7, cond_spatial], dim=1)               # [B, nf*8+512, 1, 1]
        d = self.dec1(d)                                        # [B, nf*8,  2,  2]
        d = self.dec2(torch.cat([d, e6], dim=1))               # [B, nf*8,  4,  4]
        d = self.dec3(torch.cat([d, e5], dim=1))               # [B, nf*8,  8,  8]
        d = self.dec4(torch.cat([d, e4], dim=1))               # [B, nf*8, 16, 16]
        d = self.dec5(torch.cat([d, e3], dim=1))               # [B, nf*4, 32, 32]
        d = self.dec6(torch.cat([d, e2], dim=1))               # [B, nf*2, 64, 64]
        d = self.dec7(torch.cat([d, e1], dim=1))               # [B, nf,  128,128]

        # Final output conv (no upsampling — we're already at 128×128).
        # Skip connection from enc1 doubles the channel count again.
        glyph = self.final(torch.cat([d, e1], dim=1))          # [B, 1, 128, 128]
        return glyph


# ---------------------------------------------------------------------------
# PatchDiscriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Classifies overlapping 70×70 patches as real or fake.  The discriminator
    is conditioned on one of the style reference glyphs (concatenated along
    the channel axis) to learn per-font style consistency.

    Input:
        image       : [B, 1, 128, 128]  — real or generated glyph
        style_glyph : [B, 1, 128, 128]  — one Latin reference glyph (conditioning)
    Output:
        patch_logits : [B, 1, 14, 14]   — real/fake score per 70×70 patch
    """

    def __init__(self, in_channels: int = 2, ndf: int = 64) -> None:
        super().__init__()
        # Standard PatchGAN architecture (from pix2pix paper).
        self.model = nn.Sequential(
            _conv_block(in_channels, ndf,       norm=False),   # [B, ndf,   64, 64]
            _conv_block(ndf,         ndf * 2),                  # [B, ndf*2, 32, 32]
            _conv_block(ndf * 2,     ndf * 4),                  # [B, ndf*4, 16, 16]
            _conv_block(ndf * 4,     ndf * 8, stride=1),        # [B, ndf*8, 15, 15]
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),  # [B, 1, 14, 14]
        )

    def forward(
        self,
        image: torch.Tensor,        # [B, 1, 128, 128]
        style_glyph: torch.Tensor,  # [B, 1, 128, 128]
    ) -> torch.Tensor:
        """
        Returns
        -------
        patch_logits : [B, 1, 14, 14]  (raw logits, no sigmoid)
        """
        x = torch.cat([image, style_glyph], dim=1)   # [B, 2, 128, 128]
        return self.model(x)
