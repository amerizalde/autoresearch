import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import requests
import json
import glob
import time
import subprocess
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/train.txt"
DATA_PATH = "data/train.txt"
TOKENIZER_PATH = "data/tokenizer.bin"
VOCAB_SIZE = 50257 # Default for GPT-2 style if not specified
BPE_STATS_PATH = "data/bpe_stats.bin"

def download_data():
    if not os.path랑(DATA_PATH):
        print(f"Downloading data from {DATA_URL}...")
        response = requests.get(DATA_URL)
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print("Data already exists.")

def create_tokenizer():
    # Simplified placeholder for BPE tokenizer training
    # In a real scenario, this would use tiktoken or similar
    # For this setup, we'll use a simple character-level or predefined vocab approach
    # to keep the setup "ready to run" without heavy dependencies.
    print("Training tokenizer (simplified character-level for demo)...")
    if not os.path.exists(TOKENIZER_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        with open(TOKENIZER_PATH, "w") as f:
            json.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, f)
        print(f"Tokenizer created with vocab size: {vocab_size}")
        return stoi, itos, vocab_size
    else:
        with open(TOKENIZER_PATH, "r") as f:
            data = json.load(f)
        return data["stoi"], data["itos"], data["vocab_size"]

class TinyStoriesDataset(Dataset):
    def __init__(self, text, stoi, block_size):
        self.data = [torch.tensor(stoi[c] for c in chunk) 
                     for chunk in [text[i:i+block_size] 
                                   for i in range(0, len(text), block_size)]]
        self.stoi = stoi
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        # Pad if necessary
        if len(chunk) < self.block_size:
            pad = torch.zeros(self.block_size - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def prepare_data():
    download_data()
    stoi, itos, vocab_size = create_tokenizer()
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into train/val
    n = int(0.9 * len(text))
    train_text = text[:n]
    val_text = text[n:]
    
    dataset_train = TinyStoriesDataset(train_text, stoi, block_size=256)
    dataset_val = TinyStoriesDataset(val_text, stoi, block_size=256)
    
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=16)
    
    return train_loader, val_loader, vocab_size, stoi, itos

if __name__ == "__main__":
    prepare_data()
    print("Data preparation complete.")
