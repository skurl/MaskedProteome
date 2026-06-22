# Masked Proteome Project

Creating a transformer based masking model, trained on 11 different Listeria proteomes, to test if it can learn meaningful amino-acid context from sequence alone, as a cheap, fast alternative to large general PLM's for narrow biological problems.

## Authors

- Maciej Szczesny

## Notes & Results

Trained a ~9.5M-parameter encoder-only BERT-style masked language model on 11 Listeria proteomes (25,618 unique sequences, 500 amino acid cutoff). 

**Architecture**: token embedding → RoPE positional encoding → 12 pre-norm transformer layers (d_model 256, 8 heads) → masked-residue head over 20 amino acids, using fused scaled-dot-product attention, bf16 mixed precision, EMA weights, warmup+cosine LR, and gradient accumulation. Sequences were split at the cluster level using MMseqs2 with 40% identity.

The script that was used to train the model is available in `bin/super_runner.py`. 

**Findings**: 
- Top-1 masked-residue accuracy 21.8 ± 0.04% (top-3 ~40%, top-5 ~54%), perplexity 13.1 ± 0.02 (loss 2.57) — vs a unigram-frequency baseline of 9.8% / ppl 17.5. These numbers match the ones reported by training on a much larger , UniRef50 dataset [ESM 8M citation]. 
- Biochemical class accuracy: ~38%, with predictions still in the same biochemical class: ~21%.
- Weak but significant BLOSUM62 (ρ=0.29, p<0.0001)

The key thing I learned during this excercise is definately 1. Clustering and its importance in obtaining meaningful results 2. BART-style masking, and how training based on it might result in biologically significant results (Initially, I tried masking 5 residues in a row and "mixed" masking, with both random and continuous, but the results were lower than this one, so I quickly got discouraged) 

**Diagrams:**
![Blosum Heatmap](https://github.com/skurl/MaskedProteome/blob/main/results/blosum_heatmaps.png?raw=true)

![Blosum Scatter](https://github.com/skurl/MaskedProteome/blob/main/results/blosum_scatter.png?raw=true)

![Blosum Heatmap](https://github.com/skurl/MaskedProteome/blob/main/results/per_aa_accuracy.png?raw=true)