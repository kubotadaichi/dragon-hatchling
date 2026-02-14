import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo
from model import BDH_GPU, vocab_size

# Argument Parsing
parser = argparse.ArgumentParser(description="Train BDH-GPU on Wikitext-2")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--log_interval", type=int, default=10, help="Log interval (steps)")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face Repository ID (e.g., username/repo_name)")

args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)

# Dataset Class
class ByteTextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        # Flatten all text into a single byte sequence. 
        # Using 'utf-8' encoding to get bytes, then modulo vocab_size just in case.
        text = "".join(data)
        self.data = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.long)
        self.data = self.data % vocab_size 

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        
        # Ensure chunk is long enough (for the last batch)
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([chunk, torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)])
            
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def main():
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load Data
    print("Loading Wikitext-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = dataset["train"]["text"]
    val_text = dataset["validation"]["text"]

    train_dataset = ByteTextDataset(train_text, args.seq_len)
    val_dataset = ByteTextDataset(val_text, args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = BDH_GPU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Logging
    log_file = os.path.join(args.save_dir, "training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,step,loss,time\n")

    # Training Loop
    global_step = 0
    start_time = time.time()

    print("Starting training...")
    try:
        for epoch in range(args.epochs):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for i, (x, y) in enumerate(progress_bar):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(x) # (B, T, vocab_size)
                
                loss = criterion(output.view(-1, vocab_size), y.view(-1))
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                global_step += 1
                
                if global_step % args.log_interval == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    with open(log_file, "a") as f:
                        f.write(f"{epoch},{global_step},{loss.item()},,{time.time() - start_time}\n")
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = criterion(output.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
            
            with open(log_file, "a") as f:
                f.write(f"{epoch},{global_step},,{val_loss},{time.time() - start_time}\n")

            # Save checkpoint
            checkpoint_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            if args.push_to_hub and args.hf_repo_id:
                print(f"Pushing to Hugging Face Hub: {args.hf_repo_id}...")
                try:
                    api = HfApi()
                    create_repo(repo_id=args.hf_repo_id, exist_ok=True)
                    api.upload_file(
                        path_or_fileobj=checkpoint_path,
                        path_in_repo=f"model_epoch_{epoch+1}.pt",
                        repo_id=args.hf_repo_id,
                        repo_type="model"
                    )
                    print("Upload successful!")
                except Exception as e:
                    print(f"Failed to push to HF Hub: {e}")

    except KeyboardInterrupt:
        print("Training interrupted.")

if __name__ == "__main__":
    main()
