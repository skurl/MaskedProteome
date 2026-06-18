"""
This is the most modular version of the model, also just written as a single file for simplicity.
"""


import math
import json
import random
import copy
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

# CONFIG

@dataclass
class Config:
    # paths
    # data_dir: str = "/app/data"
    # out_dir: str = "/app/outputs"
    data_dir: str = "./data"
    out_dir: str = "./outputs"
    
    # data
    length_cutoff: int = 500
    max_sequences: Optional[int] = None

    # clustering / split
    mmseqs_tsv: Optional[str] = "./outputs/clust_cluster.tsv"   # MMseqs2 *_cluster.tsv; falls back to k-mer greedy if missing
    kmer_k: int = 4
    cluster_threshold: float = 0.5
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 1234

    # masking
    mask_rate: float = 0.15

    # model
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 8
    d_ff: int = d_model*4
    dropout: float = 0.1   # MMseqs split (less redundant) overfits at 0.0 — regularise

    # training
    batch_size: int = 64
    num_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    grad_accum: int = 2         # effective batch = batch_size * grad_accum

    # step-based LR schedule (dataset-size independent)
    warmup_steps: int = 1000    # optimiser steps, not epochs
    min_lr_ratio: float = 0.1   # cosine floor: never decay below 10% of peak LR

    # exponential moving average of weights
    use_ema: bool = True
    ema_decay: float = 0.999

    log_every: int = 100        # log throughput every N optimiser steps

    seeds: Tuple[int, ...] = (42, 123, 7)
    eval_mask_seed: int = 999

def get_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   # also seeds all CUDA devices

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIALS = ["<pad>", "<mask>", "<cls>", "<unk>", "<eos>"]
PAD_IDX, MASK_IDX, CLS_IDX, UNK_IDX, EOS_IDX = 0, 1, 2, 3, 4

# LOADER

def read_fasta(path):
    """Yield sequences from one FASTA file (stdlib only — no Biopython)."""
    seq = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                    seq = []
            else:
                seq.append(line.strip())
    if seq:
        yield "".join(seq)


def load_sequences(cfg: Config) -> List[str]:
    data_dir = Path(cfg.data_dir).expanduser().resolve()
    files = sorted(f for ext in ("*.fasta", "*.fa", "*.faa") for f in data_dir.rglob(ext))
    if not files:
        raise FileNotFoundError(f"No .fasta/.fa/.faa files found in {data_dir}")
    print(f"[data] found {len(files)} FASTA file(s)")

    sequences = [s.upper().replace("*", "") for f in files for s in read_fasta(f)]
    sequences = [s for s in sequences if len(s) < cfg.length_cutoff]

    before = len(sequences)
    sequences = sorted(set(sequences))
    print(f"[data] {before} sequences loaded, {len(sequences)} after deduplication")

    if cfg.max_sequences and len(sequences) > cfg.max_sequences:
        sequences = random.Random(cfg.split_seed).sample(sequences, cfg.max_sequences)

    if not sequences:
        raise ValueError("No sequences remain after filtering.")
    return sequences


def build_vocab(seqs: List[str]):
    amino_acids = sorted(set("".join(seqs)))                 # residues present in the data
    vocab = SPECIALS + amino_acids
    stoi = {s: i for i, s in enumerate(vocab)}

    classes = [a for a in amino_acids if a in STANDARD_AA]   # predict canonical AAs only
    out_stoi = {a: i for i, a in enumerate(classes)}
    out_itos = {i: a for a, i in out_stoi.items()}

    print(f"[vocab] {len(vocab)} input tokens, {len(classes)} output classes")
    return stoi, out_stoi, out_itos, classes

# CLUSTER

def kmer_set(seq: str, k: int) -> frozenset:
    if len(seq) < k:
        return frozenset({seq})
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1))


def greedy_cluster(seqs: List[str], k: int, threshold: float) -> List[int]:
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    kmers = [kmer_set(s, k) for s in seqs]
    reps: List[int] = []
    assignment = [-1] * len(seqs)

    for idx in order:
        ks = kmers[idx]
        placed = False
        for cid, rep in enumerate(reps):
            rk = kmers[rep]
            denom = min(len(ks), len(rk))
            if denom and len(ks & rk) / denom >= threshold:
                assignment[idx] = cid
                placed = True
                break
        if not placed:
            assignment[idx] = len(reps)
            reps.append(idx)

    print(f"[cluster] {len(seqs)} sequences -> {len(reps)} clusters  (k={k}, t={threshold})")
    return assignment


def write_indexed_fasta(seqs: List[str], path: str) -> None:
    """Dump sequences with their list index as the header (>0, >1, ...), so an
    MMseqs2 cluster TSV maps straight back to positions in `seqs`."""
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    print(f"[cluster] wrote {len(seqs)} sequences to {path}")


def clusters_from_mmseqs(seqs: List[str], tsv_path: str) -> List[int]:
    """Map each sequence to a cluster id from an MMseqs2 *_cluster.tsv
    (rep_id<TAB>member_id, ids = index in `seqs` from write_indexed_fasta)."""
    rep_of = {}
    with open(tsv_path) as fh:
        for line in fh:
            rep, member = line.split()
            rep_of[int(member)] = int(rep)

    rep_to_cid, assignment = {}, []
    for i in range(len(seqs)):
        rep = rep_of.get(i, i)                       # singletons absent from tsv -> own rep
        rep_to_cid.setdefault(rep, len(rep_to_cid))
        assignment.append(rep_to_cid[rep])
    print(f"[cluster] MMseqs2 TSV -> {len(rep_to_cid)} clusters  ({tsv_path})")
    return assignment


def cluster_split(assignment, ratios, seed):
    members = defaultdict(list)
    for i, c in enumerate(assignment):
        members[c].append(i)

    cids = list(members.keys())
    random.Random(seed).shuffle(cids)
    n = len(cids)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)

    def flatten(cc):
        out = []
        for c in cc:
            out.extend(members[c])
        return out

    tr = flatten(cids[:n_tr])
    va = flatten(cids[n_tr:n_tr + n_va])
    te = flatten(cids[n_tr + n_va:])
    print(f"[split] train {len(tr)}  val {len(va)}  test {len(te)}")
    return tr, va, te

# MASKING

def mask_sequence(seq, rng, cfg, stoi, out_stoi, classes):
    ids = [CLS_IDX]
    labels = [-100]
    for aa in seq:
        tok = stoi.get(aa, UNK_IDX)
        if aa in out_stoi and rng.random() < cfg.mask_rate:
            labels.append(out_stoi[aa])
            r = rng.random()
            if r < 0.8:
                ids.append(MASK_IDX)
            elif r < 0.9:
                ids.append(stoi[rng.choice(classes)])
            else:
                ids.append(tok)
        else:
            ids.append(tok)
            labels.append(-100)
    ids.append(EOS_IDX)        # sequence-boundary marker; not a prediction target
    labels.append(-100)
    return ids, labels


class ListDataset(Dataset):
    """Thin wrapper over a list — holds raw sequences or pre-masked tensor pairs."""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


def pad_batch(items):
    xs = [x for x, _ in items]
    ys = [y for _, y in items]
    lengths = torch.tensor([len(x) for x in xs])
    x = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)
    attn = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return x, y, attn


def make_train_collate(cfg, stoi, out_stoi, classes):
    def collate(batch):
        items = []
        for seq in batch:
            ids, labels = mask_sequence(seq, random, cfg, stoi, out_stoi, classes)
            items.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            ))
        return pad_batch(items)
    return collate


def precompute_masked(seqs, seed, cfg, stoi, out_stoi, classes):
    rng = random.Random(seed)
    examples = []
    for seq in seqs:
        ids, labels = mask_sequence(seq, rng, cfg, stoi, out_stoi, classes)
        examples.append((
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        ))
    return ListDataset(examples)

# EMBEDDING

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin: (L, head_dim//2) -> (1, 1, L, head_dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

# MODEL

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # No bias — following ESM-2 convention
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        split = lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.W_q(x))
        k = split(self.W_k(x))
        v = split(self.W_v(x))

        cos, sin = self.rope(L)
        q, k = apply_rope(q, k, cos, sin)

        # padding_mask: (B, L) bool, True = attend. SDPA broadcasts (B,1,1,L) over heads + queries.
        attn_mask = padding_mask[:, None, None, :] if padding_mask is not None else None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class EncoderLayer(nn.Module):
    """Pre-norm encoder block: Norm -> Attn -> residual, Norm -> FFN -> residual"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), padding_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config, input_vocab_size: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(input_vocab_size, cfg.d_model, padding_idx=PAD_IDX)
        self.layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        # ponytail: no weight tying — output head (20 AA classes) and input embedding
        # (24-token vocab) have different sizes, so they can't share a matrix.
        self.fc = nn.Linear(cfg.d_model, num_classes, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                m.weight.data[PAD_IDX].zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src, attention_mask=None):
        # attention_mask: (B, L) bool, True = real token
        x = self.dropout(self.embed(src))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        return self.fc(x)

# Warmup sine scheduler

def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # never let the LR decay below min_lr_ratio * peak
        return max(min_lr_ratio, cosine)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# EMA

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """Swap EMA weights into the model (keeps a backup to restore)."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

# TRAINER

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    nll_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    # accumulate on-device; sync to Python once after the loop (not per batch)
    total_nll = torch.zeros((), device=device)
    total_tok = torch.zeros((), device=device)
    t1 = torch.zeros((), device=device)
    t3 = torch.zeros((), device=device)
    t5 = torch.zeros((), device=device)

    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        logits = model(x, attn)
        fl, fy = logits.reshape(-1, num_classes), y.reshape(-1)
        total_nll += nll_fn(fl, fy)
        m = fy != -100
        total_tok += m.sum()
        tgt = fy[m]
        top5 = fl[m].topk(5, dim=-1).indices
        hits = top5 == tgt.unsqueeze(-1)
        t1 += hits[:, :1].any(-1).sum()
        t3 += hits[:, :3].any(-1).sum()
        t5 += hits[:, :5].any(-1).sum()

    n = max(1, int(total_tok.item()))
    mean_nll = total_nll.item() / n
    return {
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "top1": 100.0 * t1.item() / n,
        "top3": 100.0 * t3.item() / n,
        "top5": 100.0 * t5.item() / n,
    }


def train(cfg, model, train_loader, val_loader, device, num_classes, seed=0):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg.label_smoothing)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # total optimiser steps = (updates per epoch) * epochs  -> drives step-based schedule
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.num_epochs
    sched = make_scheduler(opt, cfg.warmup_steps, total_steps, cfg.min_lr_ratio)

    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    global_step = 0
    tic = time.time()
    tok_acc = torch.zeros((), device=device)   # on-GPU; synced only at log time

    for epoch in range(cfg.num_epochs):
        model.train()
        opt.zero_grad()
        for step, (x, y, attn) in enumerate(train_loader):
            x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
            logits = model(x, attn)
            # divide loss by accum steps so gradients average rather than sum
            loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1)) / cfg.grad_accum
            loss.backward()
            tok_acc += (y != -100).sum()        # no .item() -> no per-step CUDA sync

            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(train_loader):
                clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                opt.zero_grad()
                sched.step()              # step-based: advance LR every optimiser update
                if ema is not None:
                    ema.update(model)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - tic
                    thr = tok_acc.item() / max(elapsed, 1e-6)
                    eta_min = (total_steps - global_step) * (elapsed / cfg.log_every) / 60.0
                    print(
                        f"  step {global_step:6d}/{total_steps} | "
                        f"lr {sched.get_last_lr()[0]:.2e} | "
                        f"{thr:.0f} tok/s | eta {eta_min:.1f}m"
                    )
                    tic = time.time()
                    tok_acc.zero_()

        # Evaluate on the EMA (shadow) weights when EMA is enabled
        if ema is not None:
            ema.apply_shadow(model)
        val = evaluate(model, val_loader, device, num_classes)

        improved = val["nll"] < best_val - 1e-4
        if improved:
            best_val = val["nll"]
            best_state = copy.deepcopy(model.state_dict())   # captures shadow weights if applied
            # crash/timeout insurance: best-so-far always on disk per seed
            torch.save(best_state, Path(cfg.out_dir) / f"best_seed{seed}.pth")

        if ema is not None:
            ema.restore(model)

        print(
            f"  epoch {epoch + 1:3d} | "
            f"val nll {val['nll']:.4f} | ppl {val['perplexity']:.3f} | "
            f"top1 {val['top1']:.2f}% | top3 {val['top3']:.2f}% | top5 {val['top5']:.2f}% | "
            f"lr {sched.get_last_lr()[0]:.2e} {'*' if improved else ''}"
        )

    model.load_state_dict(best_state)
    return model


# ADDITIONAL TESTS

def unigram_baseline(train_seqs, test_ds, out_stoi, out_itos):
    counts = Counter()
    for s in train_seqs:
        for aa in s:
            if aa in out_stoi:
                counts[aa] += 1
    n = sum(counts.values())
    freq = np.clip(
        np.array([counts[out_itos[i]] / n for i in range(len(out_stoi))]),
        1e-12, None,
    )
    ranking = np.argsort(-freq)

    labels = []
    for _, y in test_ds:
        labels.extend(int(t) for t in y.tolist() if t != -100)
    labels = np.array(labels)

    nll = float(-np.mean(np.log(freq[labels])))
    return {
        "top1": 100.0 * np.mean(labels == ranking[0]),
        "top3": 100.0 * np.mean(np.isin(labels, ranking[:3])),
        "top5": 100.0 * np.mean(np.isin(labels, ranking[:5])),
        "perplexity": math.exp(nll),
        "nll": nll,
    }


AA_CLASS = {
    **{a: "hydrophobic" for a in "AVILM"},
    **{a: "aromatic" for a in "FWY"},
    **{a: "polar" for a in "STNQ"},
    **{a: "basic" for a in "KRH"},
    **{a: "acidic" for a in "DE"},
    **{a: "special" for a in "GPC"},
}


@torch.no_grad()
def biochemical_breakdown(model, loader, device, out_itos):
    """Per-amino-acid accuracy + whether wrong guesses stay in the right
    biochemical class (a conservative error is more 'biologically sensible')."""
    model.eval()
    tot = defaultdict(int)
    cor = defaultdict(int)
    class_correct = same_when_wrong = wrong = 0

    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        preds = model(x, attn).argmax(-1)
        m = y != -100
        for t, p in zip(y[m].cpu().tolist(), preds[m].cpu().tolist()):
            ta, pa = out_itos[t], out_itos[p]
            tot[ta] += 1
            same_class = AA_CLASS.get(ta) == AA_CLASS.get(pa)
            if ta == pa:
                cor[ta] += 1
            else:
                wrong += 1
                if same_class:
                    same_when_wrong += 1
            if same_class:
                class_correct += 1

    total = sum(tot.values())
    per_aa = {a: {"acc": 100.0 * cor[a] / tot[a], "n": tot[a]} for a in sorted(tot)}
    return {
        "per_aa": per_aa,
        "biochemical_class_accuracy": 100.0 * class_correct / max(1, total),
        "wrong_but_same_class_rate": 100.0 * same_when_wrong / max(1, wrong),
    }


def aggregate(runs, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=0))}
    return out


def _spearman(a, b):
    # rank with average ties, then Pearson on ranks. numpy-only — avoids a scipy dep.
    def rank(x):
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        avg = (csum - cnt + csum - 1) / 2.0   # 0-based average rank per tied group
        return avg[inv]
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


@torch.no_grad()
def blosum_correlation(model, loader, device, num_classes, out_itos):
    """Does the model's confusion structure match evolutionary substitutability?

    Build a soft confusion matrix (mean predicted probability of residue j when
    the true residue is i, over all masked test positions), symmetrise it, and
    Spearman-correlate its off-diagonal entries with BLOSUM62.
    """
    from Bio.Align import substitution_matrices

    model.eval()
    conf = torch.zeros(num_classes, num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)
    for x, y, attn in loader:
        x, y, attn = x.to(device), y.to(device), attn.to(device).bool()
        probs = model(x, attn).softmax(-1).reshape(-1, num_classes)
        fy = y.reshape(-1)
        m = fy != -100
        tgt = fy[m]
        conf.index_add_(0, tgt, probs[m])
        counts.index_add_(0, tgt, torch.ones_like(tgt, dtype=conf.dtype))
    conf = (conf / counts.clamp(min=1).unsqueeze(1)).cpu().numpy()   # P(pred j | true i)
    conf = 0.5 * (conf + conf.T)                                     # symmetrise to match BLOSUM

    blosum = substitution_matrices.load("BLOSUM62")
    aas = [out_itos[i] for i in range(num_classes)]
    B = np.array([[blosum[a, b] for b in aas] for a in aas], dtype=float)

    iu = np.triu_indices(num_classes, k=1)                           # off-diagonal: substitutions only
    return {
        "spearman_offdiag": _spearman(conf[iu], B[iu]),
        "aas": aas,
        "model_matrix": conf.tolist(),
    }


# MAIN

def main():
    cfg = Config()
    device = get_device()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={device}  data={cfg.data_dir}  out={cfg.out_dir}")

    # Data, vocab and split are seed-independent (fixed split_seed / eval_mask_seed)
    # so the test set is identical across every training seed.
    seqs = load_sequences(cfg)
    stoi, out_stoi, out_itos, classes = build_vocab(seqs)
    num_classes = len(classes)

    if cfg.mmseqs_tsv and Path(cfg.mmseqs_tsv).exists():
        assignment = clusters_from_mmseqs(seqs, cfg.mmseqs_tsv)
    else:
        assignment = greedy_cluster(seqs, cfg.kmer_k, cfg.cluster_threshold)
    tr_idx, va_idx, te_idx = cluster_split(assignment, cfg.split_ratios, cfg.split_seed)
    train_seqs = [seqs[i] for i in tr_idx]
    val_seqs   = [seqs[i] for i in va_idx]
    test_seqs  = [seqs[i] for i in te_idx]

    val_ds  = precompute_masked(val_seqs,  cfg.eval_mask_seed,     cfg, stoi, out_stoi, classes)
    test_ds = precompute_masked(test_seqs, cfg.eval_mask_seed + 1, cfg, stoi, out_stoi, classes)
    val_loader  = DataLoader(val_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_batch)

    base = unigram_baseline(train_seqs, test_ds, out_stoi, out_itos)
    print(
        f"\n[baseline] unigram  top1 {base['top1']:.2f}%  top3 {base['top3']:.2f}%  "
        f"top5 {base['top5']:.2f}%  ppl {base['perplexity']:.3f}"
    )

    runs = []
    best = None  # (top1, model) — kept for biochem breakdown + checkpoint
    for seed in cfg.seeds:
        print(f"\n===== seed {seed} =====")
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            ListDataset(train_seqs),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=make_train_collate(cfg, stoi, out_stoi, classes),
            generator=g,
        )

        model = Transformer(cfg, input_vocab_size=len(stoi), num_classes=num_classes)
        if seed == cfg.seeds[0]:
            print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

        model = train(cfg, model, train_loader, val_loader, device, num_classes, seed=seed)
        tm = evaluate(model, test_loader, device, num_classes)
        tm["seed"] = seed
        print(
            f"  TEST seed {seed}  top1 {tm['top1']:.2f}%  top3 {tm['top3']:.2f}%  "
            f"top5 {tm['top5']:.2f}%  ppl {tm['perplexity']:.3f}"
        )
        runs.append(tm)
        if best is None or tm["top1"] > best[0]:
            best = (tm["top1"], model)

    agg = aggregate(runs, ["top1", "top3", "top5", "perplexity", "nll"])
    bio = biochemical_breakdown(best[1], test_loader, device, out_itos)

    print("\n================  SUMMARY  ================")
    print(f"{'unigram baseline':16s} top1 {base['top1']:6.2f}%  top3 {base['top3']:6.2f}%  "
          f"top5 {base['top5']:6.2f}%  ppl {base['perplexity']:.3f}")
    print(f"{'model (mean±std)':16s} "
          f"top1 {agg['top1']['mean']:.2f}±{agg['top1']['std']:.2f}%  "
          f"top3 {agg['top3']['mean']:.2f}±{agg['top3']['std']:.2f}%  "
          f"top5 {agg['top5']['mean']:.2f}±{agg['top5']['std']:.2f}%  "
          f"ppl {agg['perplexity']['mean']:.3f}±{agg['perplexity']['std']:.3f}")
    print(f"\n[biochem] class-accuracy {bio['biochemical_class_accuracy']:.2f}%  | "
          f"wrong-but-same-class {bio['wrong_but_same_class_rate']:.2f}%")
    for a, d in bio["per_aa"].items():
        print(f"          {a}: {d['acc']:5.1f}%  (n={d['n']})")

    out = Path(cfg.out_dir)
    torch.save(best[1].state_dict(), out / "model_best.pth")
    with open(out / "results.json", "w") as fh:
        json.dump({"baseline": base, "per_seed": runs,
                   "aggregate": agg, "biochemistry": bio}, fh, indent=2)
    print(f"\nSaved model_best.pth and results.json to {cfg.out_dir}")


if __name__ == "__main__":
    main()
