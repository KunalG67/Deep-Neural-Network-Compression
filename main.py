import os
import torch
from data import data_loader
from models import cifar_model
from compression import prune_model, quantize_model
from compression.prune import count_sparsity
from utils import train_and_eval, test_eval, save_model_npz, load_model_from_npz
from config import config_device

import kagglehub

if __name__ == '__main__':

    # ── Dataset Setup ────────────────────────────────────────────────────
    path     = kagglehub.dataset_download("moltean/fruits")
    path     = os.path.join(path, "fruits-360_100x100", "fruits-360")
    training = os.path.join(path, "Training")
    test     = os.path.join(path, "Test")

    # ── Data Loader ──────────────────────────────────────────────────────
    train_loader, test_loader = data_loader(training, test)

    # ── Verify Batch Shapes ──────────────────────────────────────────────
    batch = next(iter(train_loader))
    print("\n=== Batch Shape Verification ===")
    for k, v in batch.items():
        print(f"{k:<20} {tuple(v.shape)}")

    # ── Device + Model ───────────────────────────────────────────────────
    device = config_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # auto-tunes conv algorithms
        torch.backends.cudnn.enabled   = True
        print(f"[Config] cuDNN benchmark enabled")
    model  = cifar_model().to(device)
    print(f"\n[Model] Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ════════════════════════════════════════════════════════════════════
    # STEP 1 — Baseline Training
    # ════════════════════════════════════════════════════════════════════
    print("\n=== Step 1: Baseline Training ===")
    train_and_eval(model, train_loader, test_loader, device, epochs=10)
    os.makedirs("compressed_models", exist_ok=True)
    torch.save(model.state_dict(), "compressed_models/baseline.pth")
    print("[Saved] compressed_models/baseline.pth")

    # ════════════════════════════════════════════════════════════════════
    # STEP 2 — Pruning
    # ════════════════════════════════════════════════════════════════════
    print("\n=== Step 2: Pruning (threshold=0.05) ===")
    prune_model(model, threshold=0.05)
    count_sparsity(model)

    print("\n=== Step 2b: Retrain after Pruning ===")
    train_and_eval(model, train_loader, test_loader, device, epochs=5)
    torch.save(model.state_dict(), "compressed_models/pruned.pth")
    print("[Saved] compressed_models/pruned.pth")

    # ════════════════════════════════════════════════════════════════════
    # STEP 3 — Quantization
    # ════════════════════════════════════════════════════════════════════
    print("\n=== Step 3: Quantization (k=16 centroids) ===")
    quantize_model(model, k=16)

    print("\n=== Step 3b: Retrain after Quantization ===")
    train_and_eval(model, train_loader, test_loader, device, epochs=5)
    torch.save(model.state_dict(), "compressed_models/quantized.pth")
    print("[Saved] compressed_models/quantized.pth")

    # ════════════════════════════════════════════════════════════════════
    # STEP 4 — Save Compressed (.npz)
    # ════════════════════════════════════════════════════════════════════
    print("\n=== Step 4: Save Compressed Model (.npz) ===")
    save_model_npz(model, "compressed_models/compressed.npz")

    # ════════════════════════════════════════════════════════════════════
    # STEP 5 — Load + Final Evaluation
    # ════════════════════════════════════════════════════════════════════
    print("\n=== Step 5: Load Compressed Model + Final Eval ===")
    model2 = cifar_model().to(device)
    model2 = load_model_from_npz(
        model2, "compressed_models/compressed.npz", device
    )
    test_eval(model2, test_loader, device)

    # ════════════════════════════════════════════════════════════════════
    # STEP 6 — Size Comparison
    # ════════════════════════════════════════════════════════════════════
    print("\n=== Step 6: Model Size Comparison ===")
    def get_mb(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)

    baseline_mb   = get_mb("compressed_models/baseline.pth")
    pruned_mb     = get_mb("compressed_models/pruned.pth")
    quantized_mb  = get_mb("compressed_models/quantized.pth")
    compressed_mb = get_mb("compressed_models/compressed.npz")

    print(f"  Baseline   (.pth) : {baseline_mb:.2f} MB")
    print(f"  Pruned     (.pth) : {pruned_mb:.2f} MB  "
          f"({100*(1 - pruned_mb/baseline_mb):.1f}% smaller)")
    print(f"  Quantized  (.pth) : {quantized_mb:.2f} MB  "
          f"({100*(1 - quantized_mb/baseline_mb):.1f}% smaller)")
    print(f"  Compressed (.npz) : {compressed_mb:.2f} MB  "
          f"({100*(1 - compressed_mb/baseline_mb):.1f}% smaller)")