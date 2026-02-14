import argparse
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BDH_GPU, vocab_size, D, N, H

def evaluate(model_path, device="cuda", batch_size=8, seq_len=128, max_chunks=None):
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load Model
    model = BDH_GPU().to(device)
    try:
        if os.path.exists(model_path):
            print(f"Loading model from local path: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
        else:
            print(f"Model path {model_path} not found locally. Assuming it's a HF Repo ID.")
            from huggingface_hub import hf_hub_download
            print(f"Downloading from HF Hub: {model_path}")
            # Assuming the file name on HF is model_epoch_3.pt based on user input
            checkpoint_path = hf_hub_download(repo_id=model_path, filename="model_epoch_3.pt")
            state_dict = torch.load(checkpoint_path, map_location=device)
            
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()

    # Load Data (Validation Split)
    print("Loading Wikitext-2 validation set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "".join(dataset["text"])
    data = torch.tensor([ord(c) for c in text], dtype=torch.long) % vocab_size
    
    # Simple chunking for evaluation
    num_chunks = (len(data) - 1) // seq_len
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    if max_chunks:
        num_chunks = min(num_chunks, max_chunks)
        print(f"Limiting evaluation to {num_chunks} chunks.")
    
    print(f"Evaluating on {num_chunks} chunks...")
    with torch.no_grad():
        for i in tqdm(range(num_chunks)):
            start = i * seq_len
            end = start + seq_len + 1
            chunk = data[start:end].to(device)
            x = chunk[:-1].unsqueeze(0) # Batch size 1
            y = chunk[1:].unsqueeze(0)
            
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            
    avg_loss = total_loss / num_chunks
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    # Save Report
    with open("results/evaluation_report.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Validation Loss: {avg_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.4f}\n")

    # Weight Analysis
    print("Performing Weight Analysis...")
    weights = {
        "Encoder": model.encoder.detach().cpu().numpy().flatten(),
        "Decoder X": model.decoder_x.detach().cpu().numpy().flatten(),
        "Decoder Y": model.decoder_y.detach().cpu().numpy().flatten(),
        "Readout": model.readout.detach().cpu().numpy().flatten()
    }
    
    plt.figure(figsize=(12, 8))
    for i, (name, w) in enumerate(weights.items()):
        plt.subplot(2, 2, i+1)
        plt.hist(w, bins=100, alpha=0.7, log=True)
        plt.title(f"{name} Weight Distribution (Log Scale)")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("results/weight_distribution.png")
    print("Weight distribution plot saved to results/weight_distribution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to local model checkpont or HF Repo ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_chunks", type=int, default=None, help="Limit number of chunks for faster evaluation")
    args = parser.parse_args()
    
    evaluate(args.model_path, args.device, max_chunks=args.max_chunks)
