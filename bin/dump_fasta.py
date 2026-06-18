"""
Write the deduplicated training sequences to outputs/seqs.fasta with index
headers (>0, >1, ...), in the exact order improved.py uses — so an MMseqs2
cluster TSV maps straight back. Run once before clustering.

    singularity exec --bind $PWD:/app --pwd /app ./masked-proteome.sif \
        python -u /app/bin/dump_fasta.py
"""

from pathlib import Path
import improved as M

cfg = M.Config()
seqs = M.load_sequences(cfg)
out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
M.write_indexed_fasta(seqs, out / "seqs.fasta")
