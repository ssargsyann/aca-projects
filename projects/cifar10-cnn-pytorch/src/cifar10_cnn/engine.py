"""Shared epoch loops with sample-weighted loss and a confusion matrix."""
import torch
from torch import nn


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, count = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            summed_loss = criterion(logits, labels)
            if training:
                (summed_loss / len(labels)).backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            loss_sum += float(summed_loss.detach().cpu())
            correct += int((predicted == labels).sum())
            count += len(labels)
            confusion += torch.bincount((labels * 10 + predicted).detach().cpu(), minlength=100).reshape(10, 10)
    if not count:
        raise ValueError("Dataset/loader must contain at least one sample.")
    matrix = confusion.tolist()
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count, "confusion_matrix": matrix}
