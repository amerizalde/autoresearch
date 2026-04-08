import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import time
import sys
from torch.utils.data import DataLoader
from prepare import prepare_data, DATA_PATH, TOKENIZER_PATH

# --- CONFIGURATION (The agent should edit these!) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
block_size = 256
max_iters = 100  # Short for demo
eval_interval = 20
learning_rate = 1e-3
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
# --------------------------------------------------

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
        self.blocks = nn.Sequential(*[Block(args) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F로(F.cross_entropy(logits, targets))

        return logits, loss

# Note: The above is a pseudocode/sketch to show the structure. 
# I'll rewrite the real implementation in the next step.
