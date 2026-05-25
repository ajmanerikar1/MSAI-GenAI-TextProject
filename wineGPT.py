import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load dataset
df = pd.read_csv("archive/winemag-data-130k-v2.csv")

def extract_vintage(title):
    match = re.search(r'\b(19\d{2}|20\d{2})\b', str(title))
    return match.group(0) if match else "Unknown"

# Clean and prepare features
df['vintage'] = df['title'].apply(extract_vintage)
df['variety'] = df['variety'].fillna('Unknown Wine')
df['description'] = df['description'].fillna('')

# Construct the unified string for the decoder-only
df['training_text'] = (
    "Wine: " + df['variety'] + 
    " | Vintage: " + df['vintage'] + 
    " | Review: " + df['description'] + " <|endoftext|>"
)

df['training_text'].to_csv("wine_corpus.txt", index=False, header=False)
print(f"Sample formatted text:\n{df['training_text'].iloc[0]}")

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Key, Query, Value projections combined into one linear layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Causal mask buffer (lower triangular matrix)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values for all heads in batch and move head forward
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape to (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        # Apply causal mask: fill upper triangle with -inf so softmax zeroes it out
        att = att.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v # (B, n_heads, T, T) x (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side-by-side

        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Pre-LN variant with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x    