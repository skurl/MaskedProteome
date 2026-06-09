import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

@dataclass
class ModelArgs:
    length_cutoff: int = 100
    masking_rate: float = 0.15
    batch_size=32
    src_vocab_size = 22
    tgt_vocab_size = 22
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 4*d_model
    max_seq_length = 1000
    dropout = 0.1
    num_epochs=100
    learning_rate = 1e-3
    weight_decay = 1e-3

def run_epoch(model, loader, criterion, optimizer, device, tgt_vocab_size, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    for X_batch, y_batch, attention_mask in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        attention_mask = attention_mask.to(device).bool() # can we not make it boolian from the get go?

        output = model(X_batch, attention_mask)

        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            y_batch.contiguous().view(-1)
        )

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    tgt_vocab_size,
    num_epochs=100,
    learning_rate=1e-4,
    save_path="./outputs/weights.pth"
):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=ModelArgs.weight_decay
    )

    model = model.to(device)

    for epoch in range(num_epochs):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            tgt_vocab_size=tgt_vocab_size,
            training=True
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                tgt_vocab_size=tgt_vocab_size,
                training=False
            )

        print(
            f"Epoch: {epoch + 1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Difference: {val_loss-train_loss}"
        )

    torch.save(model.state_dict(), save_path)

    return model
