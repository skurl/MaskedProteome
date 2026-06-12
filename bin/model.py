import torch
import torch.nn as nn
import math
import copy

# Encoder-only architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, return_attention = False):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        if return_attention:
            return output, attn_probs
            
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None, return_attention = False):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        if return_attention:
            attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask, return_attention=True)
        else:
            attn_output = self.scaled_dot_product_attention(Q, K, V, mask, return_attention=False)

        output = self.W_o(self.combine_heads(attn_output))

        if return_attention:
            return output, attn_probs
        
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.GELU = nn.GELU() # changed from ReLU

    def forward(self, x):
        return self.fc2(self.GELU(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=7, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding, groups=d_model),
            nn.GELU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.local_convolution = LocalConv(d_model, kernel_size=7, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, return_attention=False):
        if return_attention:
            attn_output, attn_probs = self.self_attn(x, x, x, mask, return_attention=True)
        else:
            attn_output = self.self_attn(x, x, x, mask, return_attention=False)

        x = self.norm1(x + self.dropout(attn_output))
        conv_output = self.local_convolution(x)
        x = self.norm2(x + self.dropout(conv_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        if return_attention:
            return x, attn_probs
        
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size,tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 20)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src, attention_mask=None, return_attention=False):
        x = self.encoder_embedding(src)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        E_attention = []

        for enc_layer in self.encoder_layers:
            if return_attention:
                x, attn_probs = enc_layer(x, mask=mask, return_attention=True)
                E_attention.append(attn_probs)
            else:
                x = enc_layer(x, mask=mask, return_attention=False)
            
            if attention_mask is not None:
                x = x * attention_mask.unsqueeze(-1)

        x = self.final_norm(x)
        output = self.fc(x)

        if return_attention:
            return output, E_attention

        return output