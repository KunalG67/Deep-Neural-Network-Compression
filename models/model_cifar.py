import torch
import torch.nn as nn
from compression.linear import modified_linear
from compression.conv2d import modified_conv2d


class SmallCIFARNet(nn.Module):
    def __init__(self, num_classes=237):
        super().__init__()

        # ── CNN for LBP (1 × 100 × 100) ─────────────────────────────────
        self.lbp_cnn = nn.Sequential(
            modified_conv2d(1, 16, 3, padding=1),    # → (16, 100, 100)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → (16,  50,  50)

            modified_conv2d(16, 32, 3, padding=1),   # → (32,  50,  50)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → (32,  25,  25)

            modified_conv2d(32, 64, 3, padding=1),   # → (64,  25,  25)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → (64,  12,  12)
        )
        # Flat output: 64 × 12 × 12 = 9216

        # ── CNN for Canny (1 × 100 × 100) ───────────────────────────────
        self.canny_cnn = nn.Sequential(
            modified_conv2d(1, 16, 3, padding=1),    # → (16, 100, 100)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → (16,  50,  50)

            modified_conv2d(16, 32, 3, padding=1),   # → (32,  50,  50)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → (32,  25,  25)

            modified_conv2d(32, 64, 3, padding=1),   # → (64,  25,  25)
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → (64,  12,  12)
        )
        # Flat output: 64 × 12 × 12 = 9216

        self.flatten = nn.Flatten()

        # ── MLP: CNN outputs + 12 handcrafted features → 237 classes ────
        # 9216 (lbp) + 9216 (canny) + 6 (color) + 6 (shape) = 18444
        fused_dim = 9216 + 9216 + 6 + 6

        self.mlp = nn.Sequential(
            modified_linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            modified_linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            modified_linear(512, 256),
            nn.ReLU(),

            modified_linear(256, num_classes),        # → 237
        )

    def forward(self, batch):
        lbp         = batch["lbp"]            # (B, 1, 100, 100)
        canny       = batch["canny"]          # (B, 1, 100, 100)
        color_feats = batch["color_features"] # (B, 6)
        shape_feats = batch["shape_features"] # (B, 6)

        lbp_out   = self.flatten(self.lbp_cnn(lbp))     # (B, 9216)
        canny_out = self.flatten(self.canny_cnn(canny))  # (B, 9216)

        # Fuse CNN outputs with 12 handcrafted features
        fused = torch.cat([
            lbp_out,      # (B, 9216)
            canny_out,    # (B, 9216)
            color_feats,  # (B, 6)
            shape_feats,  # (B, 6)
        ], dim=1)         # (B, 18444)

        return self.mlp(fused)   # (B, 237)

    # ── Pruning: called from compression/prune.py ────────────────────────
    def prune(self, threshold):
        for module in self.modules():
            if isinstance(module, (modified_conv2d, modified_linear)):
                module.prune(threshold)

    # ── Quantization: called from compression/prune.py ───────────────────
    def quantize(self, k):
        for module in self.modules():
            if isinstance(module, (modified_conv2d, modified_linear)):
                module.quantize(k)


def cifar_model():
    return SmallCIFARNet(num_classes=237)