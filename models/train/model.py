"""
PyTorch model architecture for Cyrillic font glyph generation.
Conditional GAN with style encoder + U-Net generator.
"""

import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    """
    Encodes N reference glyphs into a fixed-size style vector.
    Uses shared-weight CNN + mean-pooling for permutation invariance.
    """
    def __init__(self, style_dim=256):
        super().__init__()
        self.style_dim = style_dim
        
        # Shared CNN for each glyph (128x128 -> style_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),   # 128 -> 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), # 16 -> 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1), # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, style_dim, 4, 1, 0), # 4 -> 1
        )
        
    def forward(self, style_glyphs):
        """
        Args:
            style_glyphs: [B, N, 1, 128, 128] float32 in [-1, 1]
        Returns:
            style_vector: [B, style_dim, 1, 1]
        """
        B, N, C, H, W = style_glyphs.shape
        # Flatten batch and N dimensions
        x = style_glyphs.view(B * N, C, H, W)
        # Encode each glyph
        encoded = self.encoder(x)  # [B*N, style_dim, 1, 1]
        # Reshape and mean-pool over N
        encoded = encoded.view(B, N, self.style_dim, 1, 1)
        style_vector = encoded.mean(dim=1)  # [B, style_dim, 1, 1]
        return style_vector


class UNetGenerator(nn.Module):
    """
    U-Net generator conditioned on style vector.
    Generates 128x128 glyph from character index + style.
    """
    def __init__(self, num_chars=66, style_dim=256):
        super().__init__()
        self.num_chars = num_chars
        self.style_dim = style_dim
        
        # Character embedding
        self.char_embed = nn.Embedding(num_chars, 128)
        
        # Encoder (downsampling)
        self.enc1 = self._conv_block(1, 64, norm=False)     # 128 -> 64
        self.enc2 = self._conv_block(64, 128)               # 64 -> 32
        self.enc3 = self._conv_block(128, 256)              # 32 -> 16
        self.enc4 = self._conv_block(256, 512)              # 16 -> 8
        self.enc5 = self._conv_block(512, 512)              # 8 -> 4
        
        # Bottleneck (inject style here)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + style_dim + 128, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (upsampling with skip connections)
        self.dec5 = self._upconv_block(512, 512)            # 4 -> 8
        self.dec4 = self._upconv_block(1024, 512)           # 8 -> 16 (512 + 512 skip)
        self.dec3 = self._upconv_block(1024, 256)           # 16 -> 32 (512 + 512 skip)
        self.dec2 = self._upconv_block(512, 128)            # 32 -> 64 (256 + 256 skip)
        self.dec1 = self._upconv_block(256, 64)             # 64 -> 128 (128 + 128 skip)
        
        # Final output layer
        self.out = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def _conv_block(self, in_ch, out_ch, norm=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, char_index, style_vector):
        """
        Args:
            char_index: [B] int64, indices 0-65
            style_vector: [B, style_dim, 1, 1]
        Returns:
            generated_glyph: [B, 1, 128, 128] float32 in [-1, 1]
        """
        B = char_index.size(0)
        
        # Embed character index
        char_emb = self.char_embed(char_index)  # [B, 128]
        char_emb = char_emb.view(B, 128, 1, 1).expand(-1, -1, 4, 4)
        
        # Start with blank canvas (all zeros)
        x = torch.zeros(B, 1, 128, 128, device=char_index.device)
        
        # Encoder
        e1 = self.enc1(x)      # [B, 64, 64, 64]
        e2 = self.enc2(e1)     # [B, 128, 32, 32]
        e3 = self.enc3(e2)     # [B, 256, 16, 16]
        e4 = self.enc4(e3)     # [B, 512, 8, 8]
        e5 = self.enc5(e4)     # [B, 512, 4, 4]
        
        # Inject style and character at bottleneck
        style_expanded = style_vector.expand(-1, -1, 4, 4)
        bottleneck_input = torch.cat([e5, style_expanded, char_emb], dim=1)
        b = self.bottleneck(bottleneck_input)
        
        # Decoder with skip connections
        d5 = self.dec5(b)                          # [B, 512, 8, 8]
        d4 = self.dec4(torch.cat([d5, e4], 1))     # [B, 512, 16, 16]
        d3 = self.dec3(torch.cat([d4, e3], 1))     # [B, 256, 32, 32]
        d2 = self.dec2(torch.cat([d3, e2], 1))     # [B, 128, 64, 64]
        d1 = self.dec1(torch.cat([d2, e1], 1))     # [B, 64, 128, 128]
        
        out = self.out(torch.cat([d1, x], 1))      # [B, 1, 128, 128]
        return out


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator conditioned on style glyph.
    """
    def __init__(self):
        super().__init__()
        
        # Input: [B, 2, 128, 128] (generated glyph + one style glyph)
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),               # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),             # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),            # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 1, 1),            # 16 -> 15
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 1),              # 15 -> 14
        )
        
    def forward(self, glyph, style_glyph):
        """
        Args:
            glyph: [B, 1, 128, 128] generated or real Cyrillic glyph
            style_glyph: [B, 1, 128, 128] one Latin style reference
        Returns:
            patch_logits: [B, 1, 14, 14] per-patch real/fake scores
        """
        x = torch.cat([glyph, style_glyph], dim=1)
        return self.model(x)


class FontGeneratorGAN(nn.Module):
    """
    Complete cGAN model: StyleEncoder + UNetGenerator.
    This is the wrapper used for training and ONNX export.
    """
    def __init__(self, num_chars=66, style_dim=256):
        super().__init__()
        self.style_encoder = StyleEncoder(style_dim)
        self.generator = UNetGenerator(num_chars, style_dim)
        
    def forward(self, style_glyphs, char_index):
        """
        Args:
            style_glyphs: [B, N, 1, 128, 128] float32 in [-1, 1]
            char_index: [B] int64
        Returns:
            generated_glyph: [B, 1, 128, 128] float32 in [-1, 1]
        """
        style_vector = self.style_encoder(style_glyphs)
        generated = self.generator(char_index, style_vector)
        return generated
