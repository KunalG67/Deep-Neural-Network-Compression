import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class modified_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # Mode controls behavior in forward(): 'normal' → 'prune' → 'quantize'
        self.mode = 'normal'
        
        # Pruning mask: 1 = keep weight, 0 = pruned (zeroed out)
        self.register_buffer('mask', torch.ones_like(self.weight))

    # ── Step 1: PRUNING ─────────────────────────────────────────────────
    def prune(self, threshold):
        """Zero out weights below threshold magnitude (unstructured pruning)"""
        self.mask = (torch.abs(self.weight.data) >= threshold).float()
        self.weight.data *= self.mask
        self.mode = 'prune'

    # ── Step 2: QUANTIZATION ─────────────────────────────────────────────
    def quantize(self, k):
        """
        k-means quantization: cluster weights into k centroids.
        Only non-zero (unmasked) weights are quantized.
        """
        # Get surviving weights (non-pruned)
        weights = self.weight.data.cpu().numpy()
        mask    = self.mask.cpu().numpy()

        non_zero = weights[mask == 1].reshape(-1, 1)

        if len(non_zero) < k:
            return  # Not enough weights to form k clusters

        # Fit KMeans on non-zero weights
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(non_zero)

        # Replace each weight with its cluster centroid
        labels    = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        quantized = weights.copy()
        quantized[mask == 1] = centroids[labels]

        self.weight.data = torch.tensor(
            quantized, dtype=self.weight.dtype
        ).to(self.weight.device)

        self.mode = 'quantize'

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, input):
        if self.mode == 'normal':
            return F.linear(input, self.weight, self.bias)

        elif self.mode == 'prune':
            # Apply mask to zero out pruned weights during forward pass
            masked_weight = self.weight * self.mask
            return F.linear(input, masked_weight, self.bias)

        elif self.mode == 'quantize':
            # Mask still applied — pruned weights stay zero
            masked_weight = self.weight * self.mask
            return F.linear(input, masked_weight, self.bias)