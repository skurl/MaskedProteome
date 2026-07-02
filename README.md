# Masked Proteome Project

The goal of this project is creating a transformer based masked model, trained on 11 different Listeria proteomes, to test if it can learn meaningful amino-acid context from sequence alone, as a cheap, fast alternative to large general PLM's for narrow biological problems. ESM and ProtBERT encode protein family, structure, and function from sequence alone, but are expensive to train and run. This project tests whether a lightweight, organism-specific model trained on a single genus can learn useful amino-acid context for use as a fast scoring component in larger pipelines.

# TLDR Results

A ~9.5M-parameter masked protein language model trained on a single bacterial genus (Listeria) learns amino-acid context to a perplexity of 13.1 on a strict homolog-separated test set. The model recovers part of the substitution structure of protein evolution without supervision (BLOSUM62 ρ ≈ 0.3). A local-convolution block was ablated and gave no benefit, and the perplexity ceiling is shown to be set by data volume, not architecture: a model of the same size trained on a much larger, UniRef50 dataset instead, the identical model improves to ppl 12.6 with no overfitting.

# Authors

- Maciej Szczesny

# Methods

Trained a ~9.5M-parameter encoder-only BERT-style masked language model on [11 Listeria proteomes](https://www.uniprot.org/proteomes?query=*) (25,618 unique sequences, 500 amino acid cutoff).

**Architecture**: token embedding → RoPE positional encoding → 12 pre-norm transformer layers (d_model 256, 8 heads) → masked-residue head over 20 amino acids, using fused scaled-dot-product attention, bf16 mixed precision, EMA weights, warmup+cosine LR, and gradient accumulation. Sequences were split at the cluster level using MMseqs2 with 40% identity, followed by BERT-style masking (15% of residues; 80% <mask> / 10% random / 10% kept). The model was then trained to predict the masked residues.

The script that was used to train the model is available in `bin/super_runner.py` for without local convolution, as well as `bin/localconv_ver.py` for local convolution.

# Results:

|                         | Top-1        | Top-3 | Top-5 | Perplexity  |
| ----------------------- | ------------ | ----- | ----- | ----------- |
| Unigram baseline        | 9.8%         | 25.5% | 40.0% | 17.6        |
| Local convolution model | 21.8 ± 0.04% | 40.0% | 53.6% | 13.1 ± 0.02 |

The model with local convolution doubles baseline top-1 and reduces perplexity by 25%. Validation loss (2.57) sits well above the ~2.0 threshold below which results indicate sequence leakage. This model is weaker than the [ESM2-8M](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full) (ppl ~10.5), a model of comparable size. Almost certainly that is due to the far larger training dataset.

|                | Params | Top-1 | Perplexity | BLOSUM62 ρ-value|
| -------------- | ------ | ----- | ---------- | ---------- |
| No convolution | 9.48M | 21.82 ± 0.04% | 13.124 ± 0.02 | 0.29 |
| With Local convolution | 10.30M | 21.83 ± 0.04% | 13.129 ± 0.01 | 0.32 |


Adding the convolution layer produced no measurable change while costing ~0.8M parameters and ~20% throughput. Self-attention already captures local context and can additionally model the long-range residue contacts that protein folding creates, where residues far apart in the primary structure may contact when the protein is folded.

# Diagrams:

## Without local convolution:

![Blosum Heatmap](https://github.com/skurl/MaskedProteome/blob/main/results/blosum_heatmaps.png?raw=true)

![Blosum Scatter](https://github.com/skurl/MaskedProteome/blob/main/results/blosum_scatter.png?raw=true)

![Blosum Heatmap](https://github.com/skurl/MaskedProteome/blob/main/results/per_aa_accuracy.png?raw=true)

## With Local Convolution:

![Blosum Heatmap](https://github.com/skurl/MaskedProteome/blob/main/results/blosum_heatmaps_lc.png?raw=true)

![Blosum Scatter](https://github.com/skurl/MaskedProteome/blob/main/results/blosum_scatter_lc.png?raw=true)

![Blosum Heatmap](https://github.com/skurl/MaskedProteome/blob/main/results/per_aa_accuracy_lc.png?raw=true)

As can be seen, the model learns basic biological properties, and classifies the amino acids correctly, confusing the same residue pairs that evolution confuses as well.

# Limitations
- Trained on a single genus, other bacteria is untested.

# Conclusion

The key thing I learned during this excercise is definately 1. Clustering and its importance in obtaining meaningful results 2. BART-style masking, and how training based on it might result in biologically significant results (Initially, I tried masking 5 residues in a row and "mixed" masking, with both random and continuous, but the results were lower than this one, so I quickly got discouraged). 

The initial results were based on leaky training, that when fixed, are more in line with what is reported in the literature.
