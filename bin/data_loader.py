import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from Bio import SeqIO 
import torch
from bin.main import ModelArgs
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

def loader(data_path = "./data/uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"):
    data_path = "./data/uniprotkb_taxonomy_id_1002366_2026_06_05.fasta"
    sequences = [str(record.seq) for record in SeqIO.parse(data_path, "fasta")]
    sequences = [seq for seq in sequences if len(seq) < ModelArgs.length_cutoff]

    AminoAcids = sorted(list(set("".join(sequences))))  # X is a padding, - is masking
    aa_vocab= ["-"] + ["X"]+ AminoAcids 

    aa_stoi = {s: i for i, s in enumerate(aa_vocab)}
    aa_itos = {i: s for i, s in enumerate(aa_vocab)}

    traindata, valdata, testdata = torch.utils.data.random_split(sequences, [0.80, 0.10, 0.10])


    train_loader = DataLoader(traindata, batch_size=ModelArgs.batch_size, shuffle=True, collate_fn=make_collate_fn(ModelArgs.masking_rate))
    val_loader = DataLoader(valdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=make_collate_fn(ModelArgs.masking_rate))
    test_loader = DataLoader(testdata, batch_size=ModelArgs.batch_size, shuffle=False, collate_fn=make_collate_fn(ModelArgs.masking_rate))
    
    return traindata, valdata, testdata