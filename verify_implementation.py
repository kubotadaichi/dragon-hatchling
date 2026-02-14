import torch
from model import BDH_GPU, N, D, H, vocab_size

def verify():
    print("Initializing BDH_GPU model...")
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    try:
        model = BDH_GPU()
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # Create dummy input: Batch size 2, Sequence length 16, random integers in range [0, vocab_size)
    B, T = 2, 16
    x = torch.randint(0, vocab_size, (B, T))
    
    print(f"Running forward pass with input shape {x.shape}...")
    try:
        y = model(x)
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        # Debugging hints
        print("Check dimensions in matmuls.")
        return

    expected_shape = (B, T, vocab_size)
    if y.shape == expected_shape:
        print(f"Output shape matches expected: {y.shape}")
    else:
        print(f"Output shape mismatch! Expected {expected_shape}, got {y.shape}")
        return

    if torch.isnan(y).any() or torch.isinf(y).any():
        print("Output contains NaNs or Infs!")
        return
    else:
        print("Output values are valid (no NaNs/Infs).")

    # Simple loss and backward pass
    target = torch.randint(0, vocab_size, (B, T))
    loss = torch.nn.functional.cross_entropy(y.view(-1, vocab_size), target.view(-1))
    
    print(f"Computed loss: {loss.item()}")
    
    try:
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return

    # Check gradients
    has_grads = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads = True
            if torch.isnan(param.grad).any():
                print(f"Gradient for {name} contains NaNs!")
        else:
            print(f"Warning: {name} has no gradient.")
            
    if has_grads:
        print("Gradients computed successfully.")
    else:
        print("No gradients computed.")
        return

    print("\nVerification Passed!")

if __name__ == "__main__":
    verify()
