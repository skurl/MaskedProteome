"""
Contiguous Masking Experiment (outdated model)
"""

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

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # when running locally with newer PyTorch versions with torch.accelerator support

# device = "cuda" if torch.cuda.is_available() else "cpu" # alternative for older PyTorch versions without torch.accelerator, for use on the cluster 

print(f"Using: {device} \n")
print("Masking mode: fixed contiguous span masking, span_size=3")

@dataclass
class ModelArgs:
    length_cutoff: int = 100
    masking_rate: float = 0.1
    batch_size=32
    src_vocab_size = 23
    tgt_vocab_size = 23
    d_model = 128
    num_heads = 8
    num_layers = 8
    d_ff = 4*d_model
    max_seq_length = 1000
    dropout = 0.15
    num_epochs=150
    learning_rate = 1e-3
    weight_decay = 1e-2


data_path = "./data/uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"

from pathlib import Path
from Bio import SeqIO

data_dir = Path("./data")

fasta_files = list(data_dir.glob("*.fasta")) + list(data_dir.glob("*.fa"))

sequences = []

for fasta_file in fasta_files:
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))

print("Total sequences loaded:", len(sequences))

sequences = [seq for seq in sequences if len(seq) < ModelArgs.length_cutoff]
print("Number of sequences:", len(sequences))
sequences = sorted(set(sequences), key=len)
print("Number of unique sequences:", len(sequences))

AminoAcids = sorted(list(set("".join(sequences))))  # X is a padding, - is masking
aa_vocab= ["X"]+ ["-"] + ["[CLS]"] + AminoAcids 

aa_stoi = {s: i for i, s in enumerate(aa_vocab)}
aa_itos = {i: s for i, s in enumerate(aa_vocab)}

output_stoi = {s: i for i, s in enumerate(AminoAcids)}
output_itos = {i: s for i, s in enumerate(AminoAcids)}

pad_idx = aa_stoi["X"]
mask_idx = aa_stoi["-"]
cls_idx = aa_stoi["[CLS]"]
                  
traindata, valdata, testdata = torch.utils.data.random_split(sequences, [0.80, 0.10, 0.10])

print(f"Train size: {len(traindata)}, Val size: {len(valdata)}, Test size: {len(testdata)} \n")

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F

def make_masked(seq, mask_prob=0.15, span_size=3):
    masked = list(seq)
    labels = [-100] * len(seq)

    i = 0
    start_prob = mask_prob / span_size

    while i < len(seq):
        if torch.rand(1).item() < start_prob:
            end = min(i + span_size, len(seq))

            for j in range(i, end):
                aa = seq[j]
                labels[j] = output_stoi[aa]

                r = torch.rand(1).item()

                if r < 0.8:
                    masked[j] = "-"
                elif r < 0.9:
                    masked[j] = random.choice(list(AminoAcids))
                else:
                    masked[j] = aa

            i = end
        else:
            i += 1

    return "".join(masked), torch.tensor(labels, dtype=torch.long)

def make_collate_fn(mask_rate):
    def collate_fn(batch):
        original = list(batch)

        masked_data = [make_masked(seq, mask_rate) for seq in original]

        masked = [item[0] for item in masked_data]
        y = [torch.cat([torch.tensor([-100], dtype=torch.long), item[1]]) for item in masked_data]

        x_ids = [torch.tensor([cls_idx] + [aa_stoi[aa] for aa in seq], dtype=torch.long) for seq in masked]

        lengths = torch.tensor([len(seq) for seq in x_ids], dtype=torch.long)

        x_ids = pad_sequence(x_ids, batch_first=True, padding_value=aa_stoi["X"])

        y = pad_sequence(y, batch_first=True, padding_value=-100)

        attention_mask = torch.arange(x_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

        return x_ids, y, attention_mask

    return collate_fn

train_loader = DataLoader(traindata, batch_size=ModelArgs.batch_size, shuffle=True, collate_fn=make_collate_fn(ModelArgs.masking_rate))
val_loader = DataLoader(valdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=make_collate_fn(ModelArgs.masking_rate))
test_loader = DataLoader(testdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=make_collate_fn(ModelArgs.masking_rate))

import torch
import torch.nn as nn
import math
import copy

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
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, return_attention = False):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        if return_attention:
            return output, attn_probs
            
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None, return_attention = False):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        if return_attention:
            attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask, return_attention=True)
        else:
            attn_output = self.scaled_dot_product_attention(Q, K, V, mask, return_attention=False)

        output = self.W_o(self.combine_heads(attn_output))

        if return_attention:
            return output, attn_probs
        
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
        
    def forward(self, x, mask, return_attention=False):
        if return_attention:
            attn_output, attn_probs = self.self_attn(x, x, x, mask, return_attention=True)
        else:
            attn_output = self.self_attn(x, x, x, mask, return_attention=False)

        x = self.norm1(x + self.dropout(attn_output))
        conv_output = self.local_convolution(x)
        x = self.norm2(x + self.dropout(conv_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        if return_attention:
            return x, attn_probs
        
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size,tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 20)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, attention_mask=None, return_attention=False):
        x = self.encoder_embedding(src)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        E_attention = []

        for enc_layer in self.encoder_layers:
            if return_attention:
                x, attn_probs = enc_layer(x, mask=mask, return_attention=True)
                E_attention.append(attn_probs)
            else:
                x = enc_layer(x, mask=mask, return_attention=False)
            
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        x = self.final_norm(x)
        output = self.fc(x)

        if return_attention:
            return output, E_attention

        return output


transformer = Transformer(src_vocab_size = len(aa_vocab), tgt_vocab_size=len(AminoAcids), d_model=ModelArgs.d_model, num_heads=ModelArgs.num_heads, num_layers=ModelArgs.num_layers, d_ff=ModelArgs.d_ff, max_seq_length=ModelArgs.max_seq_length, dropout=ModelArgs.dropout)

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
    save_path="./outputs/weights_span5.pth"
):
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=ModelArgs.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model = model.to(device)

    best_val_loss = float("inf")

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

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("New Best!")
            
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch: {epoch + 1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Difference: {val_loss-train_loss}, "
            f"LR: {current_lr:.6f}"
        )

    model.load_state_dict(torch.load(save_path, map_location=device))

    return model

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

transformer = train_model(
    model=transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    tgt_vocab_size=len(AminoAcids),
    num_epochs=ModelArgs.num_epochs,
    learning_rate=ModelArgs.learning_rate,
    save_path="./outputs/weights_span5.pth"
    # save_path=save_path
)
import torch
import torch.nn as nn
import torch.optim as optim

def top_k_correct(output, y_batch, k):

    mask = y_batch != -100
    topk_predictions = output.topk(k, dim=-1).indices
    correct = (topk_predictions[mask] == y_batch[mask].unsqueeze(-1)).any(dim=-1).sum().item()
    total = mask.sum().item()
    return correct, total

def test_model(model, test_loader, device, tgt_vocab_size, criterion):
    model.eval()

    total_test_loss = 0

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_positions = 0

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

            c1, total = top_k_correct(output, y_batch, k=1)
            c3, _ = top_k_correct(output, y_batch, k=3)
            c5, _ = top_k_correct(output, y_batch, k=5)

            top1_correct += c1
            top3_correct += c3
            top5_correct += c5
            total_positions += total



    avg_test_loss = total_test_loss / len(test_loader)

    top1_accuracy = top1_correct / total_positions * 100
    top3_accuracy = top3_correct / total_positions * 100
    top5_accuracy = top5_correct / total_positions * 100

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Predicted Amino Acid in the top 1 test: {top1_accuracy:.2f}%")
    print(f"Predicted Amino Acid in the top 3 test: {top3_accuracy:.2f}%")
    print(f"Predicted Amino Acid in the top 5 test: {top5_accuracy:.2f}%")

    return avg_test_loss, top1_accuracy, top3_accuracy, top5_accuracy

test_loss, top1_accuracy, top3_accuracy, top5_accuracy = test_model(
    model=transformer,
    test_loader=test_loader,
    device=device,
    tgt_vocab_size=len(AminoAcids),
    criterion=criterion
)