import torch
import torch.nn.functional as F
from torch import nn
import math

# Hyperparameters from paper/code listing
D = 256          # internal dimension
H = 4            # heads
N = 32768        # neurons
L = 6            # layers
dropout = 0.05
vocab_size = 256 # operating on bytes

class BDH_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.wte = nn.Embedding(vocab_size, D)
        self.drop = nn.Dropout(dropout)
        
        # Encoder: projects from D to N
        self.encoder = nn.Parameter(
            torch.zeros((N, D)).normal_(std=0.02)
        )
        
        # Decoder keys: projects from H*N//H to D (effectively N to D)
        self.decoder_x = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        
        # Decoder values: projects from H*N//H to D (effectively N to D)
        self.decoder_y = nn.Parameter(
            torch.zeros((H, D, N // H)).normal_(std=0.02)
        )
        
        # Readout: projects from D to vocab_size
        self.readout = nn.Parameter(
            torch.zeros((D, vocab_size)).normal_(std=0.02)
        )
        
        self.attn = LinearAttention()

    def forward(self, idx):
        B, T = idx.size() # mini-batch dimensions
        
        # Initial embedding: (B, T) -> (B, T, D) -> (B, 1, T, D) for broadcasting
        v_ast = self.ln(self.wte(idx).unsqueeze(1)) # B, 1, T, D
        
        for _ in range(L):
            # Project to neuron space: v_ast @ decoder_x -> (B, 1, T, D) @ (H, D, N//H) -> (B, H, T, N//H)
            # This is essentially getting keys for the neurons
            x = F.relu(v_ast @ self.decoder_x) # B, H, T, N//H
            
            # Linear Attention
            a_ast = self.attn(
                Q=x, 
                K=x, 
                V=v_ast
            )
            
            # Compute values for neurons: 
            # 1. Normalize attention output: self.ln(a_ast)
            # 2. Project back to neuron space: @ self.decoder_y -> (B, H, T, N//H)
            # 3. Gating/Modulation: * x
            y = F.relu(self.ln(a_ast) @ self.decoder_y) * x # B, H, T, N//H
            
            # Reshape to flatten heads: (B, H, T, N//H) -> (B, T, H * N//H) -> (B, T, N) -> (B, 1, T, N)
            # Note: The paper code does y.transpose(1, 2).reshape(B, 1, T, N)
            # Wait, (B, H, T, N//H).trp(1,2) -> (B, T, H, N//H).reshape -> (B, T, N).unsqueeze(1) -> (B, 1, T, N)
            y = y.transpose(1, 2).reshape(B, T, N).unsqueeze(1) 
            y = self.drop(y)
            
            # Start of layer with vectors x, y
            # Update v_ast: project y back to D dimension
            # y @ self.encoder -> (B, 1, T, N) @ (N, D) -> (B, 1, T, D)
            v_ast = v_ast + self.ln(y @ self.encoder) # B, 1, T, D
            v_ast = self.ln(v_ast)
            
        # Final readout
        return v_ast.squeeze(1) @ self.readout # B, T, vocab_size

class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Q, K, V):
        # Q, K: (B, H, T, N//H)
        # V: (B, 1, T, D)
        
        Qr = apply_rotary_emb(Q)
        Kr = apply_rotary_emb(K)
        
        # Linear Attention with causal mask
        # (Qr @ Kr.mT) -> (B, H, T, N//H) @ (B, H, N//H, T) -> (B, H, T, T)
        attn_weights = (Qr @ Kr.transpose(-2, -1)).tril(diagonal=-1)
        
        # (B, H, T, T) @ (B, 1, T, D) -> (B, H, T, D) ?? 
        # Wait, broadcasting (B, H, T, T) and (B, 1, T, D) might need explicit handling if we want standard matmul behavior
        # Let's check dimensions carefully.
        # attn_weights: (B, H, T, T)
        # V: (B, 1, T, D)
        # We want output (B, H, T, D)
        # The paper code says: `return (Qr @ Kr.mT).tril(diagonal=-1) @ V`
        # PyTorch matmul broadcasts batch dimensions. 
        # (B, H, T, T) @ (B, 1, T, D) -> (B, H, T, D) should work nicely.
        
        return attn_weights @ V

# RoPE Implementation
def apply_rotary_emb(x):
    # x shape: (B, H, T, head_dim) where head_dim = N // H
    # We apply RoPE to the T dimension (sequence length)
    B, H, T, head_dim = x.shape
    
    # Calculate frequencies
    # Standard RoPE implementation details
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)).to(x.device)
    t = torch.arange(T, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.einsum('i,j->ij', t, inv_freq) # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1) # (T, head_dim)
    
    # Reshape for broadcasting: (1, 1, T, head_dim)
    emb = emb.view(1, 1, T, head_dim)
    
    # Rotate
    x_rot = rotate_half(x)
    return x * emb.cos() + x_rot * emb.sin()

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
