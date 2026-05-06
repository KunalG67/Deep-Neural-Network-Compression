import torch
import torch.nn as nn


def train_model(model, train_loader, device, epochs=10, lr=1e-3):
    """Standard training loop"""
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct    = 0
        total      = 0

        for batch in train_loader:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}

            optimizer.zero_grad()
            outputs = model(batch_in)           # (B, 237)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        acc = 100.0 * correct / total
        print(f"[Train] Epoch {epoch+1}/{epochs} "
              f"| Loss: {total_loss/len(train_loader):.4f} "
              f"| Acc: {acc:.2f}%")


def evaluate(model, loader, device):
    """Evaluate accuracy on any loader"""
    model.to(device)
    model.eval()

    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in loader:
            labels   = batch["label"].to(device)
            batch_in = {k: v.to(device) for k, v in batch.items()
                        if k != "label"}

            outputs = model(batch_in)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = 100.0 * correct / total
    print(f"[Eval] Accuracy: {acc:.2f}%")
    return acc


def train_and_eval(model, train_loader, test_loader, device, epochs=10):
    """Convenience: train then evaluate"""
    train_model(model, train_loader, device, epochs=epochs)
    return evaluate(model, test_loader, device)