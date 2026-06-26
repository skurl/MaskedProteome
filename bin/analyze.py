import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # headless: write PNGs, no display on the cluster
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import improved as M           # data + analysis utilities (NOT the model)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(max_len)

    def _build(self, n):
        freqs = torch.outer(torch.arange(n, device=self.inv_freq.device).float(), self.inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, n):
        if n > self.cos.size(0):
            self._build(n)
        return self.cos[:n], self.sin[:n]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = torch.cat([cos, cos], -1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], -1).unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, padding_mask=None):
        B, L, D = x.shape
        split = lambda t: t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        cos, sin = self.rope(L)
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = padding_mask[:, None, None, :] if padding_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.W_o(out.transpose(1, 2).contiguous().view(B, L, D))


class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, conv_kernel=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = LocalConv(d_model, conv_kernel, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), padding_mask))
        h = self.norm_conv(x)
        if padding_mask is not None:
            h = h * padding_mask.unsqueeze(-1).to(h.dtype)
        x = x + self.dropout(self.conv(h))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg, input_vocab_size, num_classes):
        super().__init__()
        ck = getattr(cfg, "conv_kernel", 7)
        self.embed = nn.Embedding(input_vocab_size, cfg.d_model, padding_idx=M.PAD_IDX)
        self.layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout, ck)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, num_classes, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, src, attention_mask=None):
        x = self.dropout(self.embed(src))
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.fc(self.final_norm(x))


# --------------------------------------------------------------------------- #
# Test set + figures
# --------------------------------------------------------------------------- #

def build_test_loader(cfg, stoi, out_stoi, classes):
    seqs = M.load_sequences(cfg)
    if cfg.mmseqs_tsv and Path(cfg.mmseqs_tsv).exists():
        assignment = M.clusters_from_mmseqs(seqs, cfg.mmseqs_tsv)
    else:
        assignment = M.greedy_cluster(seqs, cfg.kmer_k, cfg.cluster_threshold)
    _, _, te_idx = M.cluster_split(assignment, cfg.split_ratios, cfg.split_seed)
    test_seqs = [seqs[i] for i in te_idx]
    test_ds = M.precompute_masked(test_seqs, cfg.eval_mask_seed + 1, cfg, stoi, out_stoi, classes)
    return DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=M.pad_batch)


def class_order(aas):
    groups = ["hydrophobic", "aromatic", "polar", "basic", "acidic", "special"]
    return sorted(range(len(aas)), key=lambda i: groups.index(M.AA_CLASS.get(aas[i], "special")))


def plot_scatter(model_mat, B, aas, rho, out):
    iu = np.triu_indices(len(aas), k=1)
    x, y = B[iu], model_mat[iu]
    plt.figure(figsize=(5.5, 5))
    plt.scatter(x, y, s=18, alpha=0.7)
    plt.xlabel("BLOSUM62 score (evolutionary substitutability)")
    plt.ylabel("model P(predict j | true i), symmetrised")
    plt.title(f"Model confusions vs BLOSUM62\noff-diagonal Spearman = {rho:.3f}")
    plt.tight_layout()
    plt.savefig(out / "blosum_scatter.png", dpi=150)
    plt.close()


def plot_heatmaps(model_mat, B, aas, out):
    order = class_order(aas)
    labels = [aas[i] for i in order]
    Mh = model_mat[np.ix_(order, order)].copy()
    Bh = B[np.ix_(order, order)].copy()
    np.fill_diagonal(Mh, np.nan)        # mask self-prediction so the colour scale shows substitutions
    np.fill_diagonal(Bh, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, mat, title in [(axes[0], Mh, "Model confusion (off-diagonal)"),
                           (axes[1], Bh, "BLOSUM62 (off-diagonal)")]:
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("AAs grouped by biochemical class — shared block structure = learned chemistry")
    plt.tight_layout()
    plt.savefig(out / "blosum_heatmaps.png", dpi=150)
    plt.close()


def plot_per_aa(bio, out):
    items = sorted(bio["per_aa"].items(), key=lambda kv: -kv[1]["acc"])
    aas = [k for k, _ in items]
    acc = [v["acc"] for _, v in items]
    plt.figure(figsize=(8, 4))
    plt.bar(aas, acc)
    plt.ylabel("top-1 accuracy (%)")
    plt.xlabel("true amino acid")
    plt.title("Per-amino-acid masked-prediction accuracy")
    plt.tight_layout()
    plt.savefig(out / "per_aa_accuracy.png", dpi=150)
    plt.close()


def main():
    cfg = M.Config()
    device = M.get_device()
    out = Path(cfg.out_dir)
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else out / "model_best.pth"
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}  (pass a path as the first arg)")

    print(f"[setup] device={device}  weights={weights}")

    # Vocab is derived from the data, same as training.
    seqs_for_vocab = M.load_sequences(cfg)
    stoi, out_stoi, out_itos, classes = M.build_vocab(seqs_for_vocab)
    num_classes = len(classes)

    test_loader = build_test_loader(cfg, stoi, out_stoi, classes)

    model = Transformer(cfg, input_vocab_size=len(stoi), num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    # Sanity: confirm loaded weights reproduce sensible metrics.
    metrics = M.evaluate(model, test_loader, device, num_classes)
    print(f"[test] top1 {metrics['top1']:.2f}%  top3 {metrics['top3']:.2f}%  "
          f"top5 {metrics['top5']:.2f}%  ppl {metrics['perplexity']:.3f}")

    bio = M.biochemical_breakdown(model, test_loader, device, out_itos)
    blosum = M.blosum_correlation(model, test_loader, device, num_classes, out_itos)
    rho = blosum["spearman_offdiag"]

    print(f"\n[biochem] class-accuracy {bio['biochemical_class_accuracy']:.2f}%  | "
          f"wrong-but-same-class {bio['wrong_but_same_class_rate']:.2f}%")
    print(f"[blosum]  off-diagonal Spearman vs BLOSUM62: {rho:.3f}")

    # Figures
    from Bio.Align import substitution_matrices
    aas = blosum["aas"]
    model_mat = np.array(blosum["model_matrix"])
    bl = substitution_matrices.load("BLOSUM62")
    B = np.array([[bl[a, b] for b in aas] for a in aas], dtype=float)

    plot_scatter(model_mat, B, aas, rho, out)
    plot_heatmaps(model_mat, B, aas, out)
    plot_per_aa(bio, out)
    print(f"\nWrote blosum_scatter.png, blosum_heatmaps.png, per_aa_accuracy.png to {out}")


if __name__ == "__main__":
    main()
