import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# To-Do:
# PyTorch Architecture
# Train
# Save Weights
# Evaluate performance/fine tune
# Chat bot

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

# Byte Pair Tokenization

# Initialize an empty BPE model
tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

# Configure the trainer
# Target a vocabulary size of 5,000 tokens for WineGPT
trainer = BpeTrainer(
    vocab_size=5000, 
    special_tokens=["<|endoftext|>"]
)

# Train on local wine text file
tokenizer.train(["wine_corpus.txt"], trainer)
tokenizer.save("wine_gpt_tokenizer.json")

# Tokenizer test
encoded = tokenizer.encode("Wine: Pinot Noir | Vintage: 2018 <|endoftext|>")
print(f"\nToken IDs: {encoded.ids}")
print(f"Decoded Text: {tokenizer.decode(encoded.ids)}")

# PyTorch Model

class SelfAttention(nn.Module):
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
        self.attn = SelfAttention(d_model, n_heads, block_size)
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

class WineGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()
        self.block_size = block_size
        
        # Token and learnable positional embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        
        # Stacked Decoder layers
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size) for _ in range(n_layers)
        ])
        
        # Final LayerNorm before projecting to vocabulary space
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Standard initialization for weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Token and positional embeddings are added together
        tok_emb = self.token_embedding_table(idx) # Shape: (B, T, d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # Shape: (T, d_model)
        x = tok_emb + pos_emb # Shape: (B, T, d_model)
        
        # Pass through the transformer blocks
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) # Shape: (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape tensors for PyTorch's cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss



class WineDataset(Dataset):
    def __init__(self, df, tokenizer, block_size):
        print("Tokenizing entire wine corpus into memory...")
        # Uses the HF tokenizer implementation to rapidly tokenize all rows quickly
        encodings = tokenizer.encode_batch(df['training_text'].tolist())
        all_tokens = []
        for encoding in encodings:
            all_tokens.extend(encoding.ids)
            
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # Avoid 1-token step sliding window which leads to 128x redundancy and extremely slow training.
        # Instead, divide into non-overlapping blocks of size block_size.
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        # Non-overlapping chunk retrieval
        start_idx = idx * self.block_size
        x = self.data[start_idx : start_idx + self.block_size]
        y = self.data[start_idx + 1 : start_idx + self.block_size + 1]
        return x, y

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 128     # Max context length for our wine reviews
D_MODEL = 256        # Embedding dimension
N_HEADS = 4          # Number of attention heads
N_LAYERS = 4         # Number of transformer blocks
LEARNING_RATE = 3e-4
EPOCHS = 1           # Adjust based on compute availability
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Vocab Size from tokenizer
VOCAB_SIZE = tokenizer.get_vocab_size()

# Instantiations
dataset = WineDataset(df, tokenizer, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = WineGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
model.train()
print(f"Training on device: {DEVICE}...")

for epoch in range(EPOCHS):
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

# Save the trained model weights
model_path = "wine_gpt_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Training completed successfully. Saved model weights to '{model_path}'.")