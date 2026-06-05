import torch
import torch.nn as nn
import torch.optim as optim

def run_epoch(model, loader, criterion, optimizer, device, tgt_vocab_size, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()

        output = model(X_batch)

        if training:
            optimizer.zero_grad()

        tgt_output = y_batch

        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_output.contiguous().view(-1)
        )

        if training:
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
    learning_rate=0.0001,
    save_path="./outputs/weights.pth"
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
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