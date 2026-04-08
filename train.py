import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time
import sys
import argparse
from torch.utils.data import DataLoader
from prepare import prepare_data, DATA_PATH, TOKENIZER_PATH

# --- CONFIGURATION (The agent should edit these!) ---
# These variables are placed at the top level for easy identification by the agent.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
block_size = 256
max_iters = 100  
eval_interval = 20
learning_rate = 1e-3
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
vocab_size = 0 # Will be updated by prepare_data
stoi = {}
itos = {}
# ----------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.key = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.query = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.Ch = args.n_embd
        self.heads = args.n_head
        self.dropout = args.dropout
        self.register_buffer('bias', torch.tril(torch.ones(args.block_size, args.block_size))
                                   .view(1, 1, args.block_size, args.block_size))

        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.dropout_layer = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.shape[-1] ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout_layer(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.proj(y)
        y = self.dropout_layer(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd),
            nn.ReLU(),
            nn.Linear(4 * args.n_embd, args.n_embd),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sa = MultiHeadAttention(args)
        self.ffwd = FeedForward(args)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.token_embedding_table = nn.Embedding(args.vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args) for _ in in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class Args:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, batch_size, learning_rate):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate

def run_training():
    global vocab_size, stoi, itos
    # Load tokenizer info
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "r") as f:
            data = json.load(f)
            stoi = data["stoi"]
            itos = data["itos"]
            vocab_size = data["vocab_size"]
    else:
        print("Tokenizer not found. Please run prepare.py first.")
        return

    # Load dataloaders
    train_loader, val_loader, _, _, _ = prepare_data()
    
    args = Args(vocab_size, n_embd, n_head, n_layer, block_size, dropout, batch_size, learning_rate)
    model = GPTLanguageModel(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Model initialized with vocab_size={vocab_size}")
    
    start_time = time.time()
    
    for iter in range(max_iters):
        # Evaluation phase
        if iter % eval_interval == 0:
            model.eval()
            total_loss = 0
            val_samples = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    total_loss += loss.item()
                    val_samples += x.size(0)
            avg_val_loss = total_loss / len(val_loader)
            # Converting loss to bits per byte (approximate for demo)
            # In a real scenario, this would be properly calculated with bits per character/byte
            val_bpb = torch.exp(torch.tensor(avg_val_loss)).item() 
            print(f"step {iter}: val_loss {avg_val_loss:.4f} (approx bpb: {val_bpb:.4f})")
            
            # Check for 5-minute limit (wall clock)
            if time.time() - start_time > 300: # 5 minutes
                print("Time limit reached.")
                break

        # Training phase
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # We only need one batch for this loop as we use max_iters
            break 
            
    print(f"Training finished. Final val_loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    run_training()
