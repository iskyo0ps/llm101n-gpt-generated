import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Example
Q = torch.rand(1, 5, 64)  # (batch_size, seq_len, d_k)
K = torch.rand(1, 5, 64)
V = torch.rand(1, 5, 64)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print("Output:", output)
print("Attention Weights:", attention_weights)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]

# Example
d_model = 64
pos_encoder = PositionalEncoding(d_model)
input_tensor = torch.zeros(1, 5, d_model)
encoded_tensor = pos_encoder(input_tensor)
print("Positional Encoded Tensor:", encoded_tensor)

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
    
    def forward(self, x):
        x = self.positional_encoding(x)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)
        return attention_output, attention_weights

# Example
seq_len = 5
d_model = 64
x = torch.rand(1, seq_len, d_model)

self_attention = SelfAttention(d_model)
output, attention_weights = self_attention(x)
print("Self-Attention Output:", output)
print("Attention Weights:", attention_weights)