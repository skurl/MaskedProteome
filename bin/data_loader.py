import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from Bio import SeqIO 
import torch

def load_and_encode(data_path = "./data/uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"):
    data_path = "./data/uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"
    sequences = [str(record.seq) for record in SeqIO.parse(source, "fasta")]
    sequences = [seq for seq in sequences if len(seq) < 1000]

    AminoAcids = sorted(list(set("".join(sequences))) + ["-"] + ["X"])

    aa_stoi = {s: i for i, s in enumerate(AminoAcids)}
    aa_itos = {i: s for i, s in enumerate(AminoAcids)}

    aa_encode = lambda s: F.one_hot(torch.tensor([aa_stoi[c] for c in s], dtype=torch.long), num_classes=len(AminoAcids)).float()
    aa_decode = lambda x: "".join([aa_itos[i] for i in x.argmax(dim=-1).tolist()])

    encoded_sequences = [aa_encode(seq) for seq in sequences]
    
    return encoded_sequences, AminoAcids,, aa_stoi, aa_itos, aa_encode, aa_decode

