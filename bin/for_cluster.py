import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from Bio import SeqIO 
import torch
import random
from dataclasses import dataclass
from pathlib import Path

random.seed(42)
np.random.seed (42)

print ("imported")

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # when running locally with newer PyTorch versions with torch.accelerator support

device = "cuda" if torch.cuda.is_available() else "cpu" # alternative for older PyTorch versions without torch.accelerator, for use on the cluster 

print(f"Using: {device} \n")

@dataclass
class ModelArgs:
    length_cutoff: int = 1000
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

import os

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

save_path = output_dir / "weights.pth"


sequences = [str(record.seq) for record in SeqIO.parse(data_path, "fasta")]
sequences = [seq for seq in sequences if len(seq) < ModelArgs.length_cutoff]

print("Number of sequences:", len(sequences))

AminoAcids = sorted(list(set("".join(sequences))))  # X is a padding, - is masking
aa_vocab= ["-"] + ["X"]+ AminoAcids 
print(len(aa_vocab))
print(aa_vocab)

aa_stoi = {s: i for i, s in enumerate(aa_vocab)}
aa_itos = {i: s for i, s in enumerate(aa_vocab)}

traindata, valdata, testdata = torch.utils.data.random_split(sequences, [0.80, 0.10, 0.10])

print(f"Train size: {len(traindata)}, Val size: {len(valdata)}, Test size: {len(testdata)} \n")

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F

def make_masked(seq, mask_prob=0.15):
    masked = []
    labels = []

    for aa in seq:
        if torch.rand(1).item() < mask_prob:
            labels.append(aa_stoi[aa])

            r = torch.rand(1).item()

            if r < 0.8:
                masked.append("-")
            elif r < 0.9:
                masked.append(random.choice(list(AminoAcids)))
            else:
                masked.append(aa)
        else:
            masked.append(aa)
            labels.append(-100)

    return "".join(masked), torch.tensor(labels, dtype=torch.long)

def make_collate_fn(mask_rate):
    def collate_fn(batch):
        original = list(batch)

        masked_data = [make_masked(seq, mask_rate) for seq in original]

        masked = [item[0] for item in masked_data]
        y = [item[1] for item in masked_data]

        x_ids = [torch.tensor([aa_stoi[aa] for aa in seq], dtype=torch.long) for seq in masked]

        lengths = torch.tensor([len(seq) for seq in x_ids], dtype=torch.long)

        x_ids = pad_sequence(x_ids, batch_first=True, padding_value=aa_stoi["X"])

        y = pad_sequence(y, batch_first=True, padding_value=-100)

        attention_mask = torch.arange(x_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

        return x_ids, y, attention_mask

    return collate_fn

train_loader = DataLoader(traindata, batch_size=ModelArgs.batch_size, shuffle=True, collate_fn=make_collate_fn(ModelArgs.masking_rate))
val_loader = DataLoader(valdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=make_collate_fn(ModelArgs.masking_rate))
test_loader = DataLoader(testdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=make_collate_fn(ModelArgs.masking_rate))

from model import Transformer

transformer = Transformer(src_vocab_size = len(aa_vocab), tgt_vocab_size=len(aa_vocab), d_model=ModelArgs.d_model, num_heads=ModelArgs.num_heads, num_layers=ModelArgs.num_layers, d_ff=ModelArgs.d_ff, max_seq_length=ModelArgs.max_seq_length, dropout=ModelArgs.dropout)

import torch
import torch.nn as nn
import torch.optim as optim

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
    save_path="/app/outputs/weights.pth"
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


criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

transformer = train_model(
    model=transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    tgt_vocab_size=len(aa_vocab),
    num_epochs=ModelArgs.num_epochs,
    learning_rate=ModelArgs.learning_rate,
    save_path="/app/outputs/weights.pth"
)

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

test_loss, test_accuracy = test_model(
    model=transformer,
    test_loader=test_loader,
    device=device,
    tgt_vocab_size=ModelArgs.tgt_vocab_size,
    criterion=criterion
)