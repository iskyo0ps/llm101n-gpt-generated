### Key Components

# 1. **Self-Attention Mechanism**
# 2. **Residual Connections**
# 3. **Layer Normalization**
# 4. **Feed-Forward Neural Network**
# 5. **Transformer Decoder Layer (GPT-2 Style)**

### Implementation
#### 1. Self-Attention Mechanism

import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


#### 2. Residual Connections and Layer Normalization

import torch.nn as nn

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super(ResidualLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=epsilon)

    def forward(self, x, sublayer):
        return self.layer_norm(x + sublayer(x))


#### 3. Feed-Forward Neural Network
class FeedForwardNN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


#### 4. Transformer Decoder Layer (GPT-2 Style)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.residual_layernorm1 = ResidualLayerNorm(d_model)
        self.feed_forward = FeedForwardNN(d_model, d_ff)
        self.residual_layernorm2 = ResidualLayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        x = self.residual_layernorm1(x, lambda x: self.self_attn(x, x, x)[0])
        # Feed-forward
        x = self.residual_layernorm2(x, self.feed_forward)
        return x


#### 5. GPT-2 Model


class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, max_len=512):
        super(GPT2, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, d_ff)
        for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return self.fc(x)

# Hyperparameters
vocab_size = 30522  # GPT-2 vocabulary size
d_model = 768       # GPT-2 hidden size
nhead = 12          # GPT-2 number of attention heads
d_ff = 3072         # GPT-2 feed-forward network hidden size
num_layers = 12     # GPT-2 number of decoder layers

# Instantiate the model
model = GPT2(vocab_size, d_model, nhead, d_ff, num_layers)

# Example input: a batch of tokenized sequences
input_ids = torch.randint(0, vocab_size, (2, 512))  # Batch size of 2, sequence length of 512
output = model(input_ids)
print("Model output shape:", output.shape)