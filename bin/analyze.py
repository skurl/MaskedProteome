"""
Loads saved weights (default outputs/model_best.pth), rebuilds the *same*
test set as training (deterministic: fixed split_seed / eval_mask_seed in
Config), and writes figures + numbers for:
  - BLOSUM62 correlation (does the model's confusion match evolution?)
  - per-amino-acid accuracy
  - model confusion matrix vs BLOSUM62 heatmaps

Run:
    singularity exec --nv --bind $PWD:/app --pwd /app ./masked-proteome.sif \
        python -u /app/bin/analyze.py [path/to/weights.pth] 2>&1 | tee analyze.log
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

matplotlib.use("Agg")          # headless: write PNGs, no display on the cluster

import improved as M


def build_test_loader(cfg, stoi, out_stoi, classes):
    """Rebuild the exact test split + masks training used (all seeds fixed)."""
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
    """Order AAs by biochemical class so block structure is visible in heatmaps."""
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
    np.fill_diagonal(Mh, np.nan)
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

    model = M.Transformer(cfg, input_vocab_size=len(stoi), num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    metrics = M.evaluate(model, test_loader, device, num_classes)
    print(f"[test] top1 {metrics['top1']:.2f}%  top3 {metrics['top3']:.2f}%  "
          f"top5 {metrics['top5']:.2f}%  ppl {metrics['perplexity']:.3f}")

    bio = M.biochemical_breakdown(model, test_loader, device, out_itos)
    blosum = M.blosum_correlation(model, test_loader, device, num_classes, out_itos)
    rho = blosum["spearman_offdiag"]

    print(f"\n[biochem] class-accuracy {bio['biochemical_class_accuracy']:.2f}%  | "
          f"wrong-but-same-class {bio['wrong_but_same_class_rate']:.2f}%")
    print(f"[blosum]  off-diagonal Spearman vs BLOSUM62: {rho:.3f}")

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
