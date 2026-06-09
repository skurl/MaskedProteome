import torch
import torch.nn as nn
import torch.optim as optim

def test_model(model, test_loader, device, tgt_vocab_size, criterion):
    model.eval()

    total_correct = 0
    total_positions = 0
    total_test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch, attention_mask in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            attention_mask = attention_mask.to(device).bool()

            output = model(X_batch, attention_mask)

            loss = criterion(
                output.contiguous().view(-1, tgt_vocab_size),
                y_batch.contiguous().view(-1)
            )

            total_test_loss += loss.item()

            predicted = output.argmax(dim=-1)

            mask = y_batch != -100

            total_correct += (predicted[mask] == y_batch[mask]).sum().item()
            total_positions += mask.sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / total_positions * 100

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Masked Test Accuracy: {test_accuracy:.2f}%")

    return avg_test_loss, test_accuracy