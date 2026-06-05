import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data_loader import load_data
from train import train_model
from model import Transformer
from testing import test_model

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # when running locally with newer PyTorch versions with torch.accelerator support

# device = "cuda" if torch.cuda.is_available() else "cpu" # alternative for older PyTorch versions without torch.accelerator, for use on the cluster 

print(f"Using: {device} \n")

X, y, AminoAcids, SecondaryStructures, aa_stoi, aa_itos, aa_encode, aa_decode, ss_stoi, ss_itos, ss_encode, ss_decode, set_length = load_data(set_length=72)

dataset = TensorDataset(X, y)

traindata, valdata, testdata = torch.utils.data.random_split(dataset, [0.80, 0.10, 0.10])

print(f"Train size: {len(traindata)}, Val size: {len(valdata)}, Test size: {len(testdata)} \n")

train_loader = DataLoader(traindata, batch_size=32, shuffle=True)
val_loader = DataLoader(valdata, batch_size=32, shuffle=False)
test_loader = DataLoader(testdata, batch_size=32, shuffle=False)

src_vocab_size = len(AminoAcids)
tgt_vocab_size = len(SecondaryStructures)
d_model = 64
num_heads = 2
num_layers = 3
d_ff = 4*d_model
max_seq_length = set_length
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

transformer = train_model(
    model=transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    tgt_vocab_size=tgt_vocab_size,
    num_epochs=300,
    learning_rate=1e-4,
    save_path="./outputs/weights.pth"
    # save_path=save_path
)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

test_loss, test_accuracy = test_model(
    model=transformer,
    test_loader=test_loader,
    device=device,
    tgt_vocab_size=tgt_vocab_size,
    criterion=criterion
)