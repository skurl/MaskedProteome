"""
Original file I would put on the cluster (outdated now, since Ive changed so many things)
"""

import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from Bio import SeqIO 
import torch
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
from pathlib import Path

random.seed(42)
np.random.seed (42)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # when running locally with newer PyTorch versions with torch.accelerator support

# device = "cuda" if torch.cuda.is_available() else "cpu" # alternative for older PyTorch versions without torch.accelerator, for use on the cluster 

print(f"Using: {device} \n")

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


BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"

sequences = [str(record.seq) for record in SeqIO.parse(data_path, "fasta")]
sequences = [seq for seq in sequences if len(seq) < ModelArgs.length_cutoff]

AminoAcids = sorted(list(set("".join(sequences))) + ["-"])

aa_stoi = {s: i for i, s in enumerate(AminoAcids)}
aa_itos = {i: s for i, s in enumerate(AminoAcids)}

traindata, valdata, testdata = torch.utils.data.random_split(sequences, [0.80, 0.10, 0.10])

print("Number of sequences:", len(sequences))
print(f"Train size: {len(traindata)}, Val size: {len(valdata)}, Test size: {len(testdata)} \n")

def collate_fn(batch):
    original = batch
    masked = ["".join("-" if torch.rand(1).item() < ModelArgs.masking_rate else aa for aa in seq) for seq in original]

    X = [torch.tensor([aa_stoi[aa] for aa in seq]) for seq in masked]

    y = [torch.tensor([aa_stoi[aa] if m == "-" else -100 for aa, m in zip(orig, mask)]) for orig, mask in zip(original, masked)]

    lengths = torch.tensor([len(seq) for seq in X])

    X = pad_sequence(X, batch_first=True, padding_value=aa_stoi["-"])
    y = pad_sequence(y, batch_first=True, padding_value=-100)

    X = F.one_hot(X, num_classes=len(aa_stoi)).float()

    attention_mask = torch.arange(X.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

    X[~attention_mask] = 0.0

    return X, y, attention_mask

train_loader = DataLoader(traindata, batch_size=ModelArgs.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(valdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(testdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=collate_fn)

# Transformer
import torch
import torch.nn as nn
import math

# Encoder-only architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.GELU = nn.GELU() # changed from ReLU

    def forward(self, x):
        return self.fc2(self.GELU(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding, groups=d_model),
            nn.GELU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.local_convolution = LocalConv(d_model, kernel_size=7, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        conv_output = self.local_convolution(x)
        x = self.norm2(x + self.dropout(conv_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size,tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.linear = nn.Linear(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, attention_mask=None):
        x = self.linear(src)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        for enc_layer in self.encoder_layers:
            x = enc_layer(x, mask=mask)

            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        x = self.final_norm(x)
        output = self.fc(x)

        return output

transformer = Transformer(ModelArgs.src_vocab_size, ModelArgs.tgt_vocab_size, ModelArgs.d_model, ModelArgs.num_heads, ModelArgs.num_layers, ModelArgs.d_ff, ModelArgs.max_seq_length, ModelArgs.dropout)

# Train loop
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
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()
        attention_mask = attention_mask.to(device).float()

        output = model(X_batch, attention_mask)

        if training:
            optimizer.zero_grad()

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
    learning_rate=0.0001,
    save_path=BASE_DIR / "outputs" / "weights.pth"
):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2
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
    tgt_vocab_size=ModelArgs.tgt_vocab_size,
    num_epochs=ModelArgs.num_epochs,
    learning_rate=ModelArgs.learning_rate,
    save_path=BASE_DIR / "outputs" / "weights.pth"
    # save_path=save_path
)

def test_model(model, test_loader, device, tgt_vocab_size, criterion):
    model.eval()

    total_correct = 0
    total_positions = 0
    total_test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch, attention_mask in test_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).long()
            attention_mask = attention_mask.to(device)

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