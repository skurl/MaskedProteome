for : 
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
    num_epochs=100
    learning_rate = 1e-3
    weight_decay = 1e-2


Test Loss: 2.1235
Predicted Amino Acid in the top 1 test: 38.05%
Predicted Amino Acid in the top 3 test: 58.27%
Predicted Amino Acid in the top 5 test: 69.97%


for : 


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
    num_epochs=130
    learning_rate = 1e-3
    weight_decay = 1e-2


Test Loss: 2.0381
Predicted Amino Acid in the top 1 test: 42.51%
Predicted Amino Acid in the top 3 test: 60.36%
Predicted Amino Acid in the top 5 test: 70.62%



ADDED label smoothing

Test Loss: 2.0123
Predicted Amino Acid in the top 5 test: 71.12%
Predicted Amino Acid in the top 1 test: 42.65%
Predicted Amino Acid in the top 3 test: 61.19%

increased #epochs to 150

Test Loss: 1.9450
Predicted Amino Acid in the top 1 test: 44.12%
Predicted Amino Acid in the top 3 test: 60.93%
Predicted Amino Acid in the top 5 test: 70.19%

saves the best epoch:

Test Loss: 1.9629
Predicted Amino Acid in the top 1 test: 41.68%
Predicted Amino Acid in the top 3 test: 60.69%
Predicted Amino Acid in the top 5 test: 70.69%

So basically now its quite consistent