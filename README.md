# Dragon Hatchling (BDH) Implementation

This repository contains a PyTorch implementation of the **Dragon Hatchling (BDH)** model, based on the paper ["Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"](https://arxiv.org/abs/2509.26507).

## Overview

The Dragon Hatchling model bridges the gap between modern Transformer architectures and biological neural networks. This implementation focuses on the **BDH-GPU** variant, which is optimized for tensor processing units.

**Key Features:**
*   **BDH-GPU Architecture:** Tensor-friendly implementation of the biological model.
*   **Linear Attention:** Efficient $O(TK)$ attention mechanism.
*   **Rotary Positional Embeddings (RoPE):** Enhanced positional encoding.
*   **Hugging Face Hub Integration:** Seamless model uploading and downloading.

## Installation

This project uses `uv` for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/kubotadaichi/dragon-hatchling.git
cd dragon-hatchling

# Install dependencies
uv sync
```

## Usage

### Training

To train the model on the Wikitext-2 dataset:

```bash
uv run experiment.py --epochs 3 --batch_size 8 --device cuda
```

**Arguments:**
*   `--push_to_hub`: Upload checkpoints to Hugging Face Hub.
*   `--hf_repo_id`: The repository ID to upload to (e.g., `username/dragon-hatchling`).

### Evaluation

To evaluate a trained model (local or from HF Hub):

```bash
uv run evaluate.py --model_path <path-or-repo-id>
```

Example:
```bash
uv run evaluate.py --model_path daichi202/dragon-hatchling --max_chunks 100
```

## Implementation Report

### Implementation Details
We implemented the GPU-friendly version of the model, **BDH-GPU**, as described in Appendix E of the paper.

#### Key Components
- **BDH-GPU Architecture:** Implemented in `model.py`.
- **Linear Attention:** Implemented with `O(TK)` complexity.
- **Rotary Positional Embeddings (RoPE):** Added to enhance positional awareness.
- **Training Script:** `experiment.py` handles data loading (Wikitext-2), training loop, and Hugging Face Hub integration.

### Experimental Setup
- **Dataset:** Wikitext-2 (Byte-level tokenization)
- **Model Parameters:**
    - Hidden Dimension ($D$): 256
    - Heads ($H$): 4
    - Neurons ($N$): 32768
    - Layers ($L$): 2
    - Vocab Size: 256
- **Training:**
    - Epochs: 3
    - Batch Size: 8
    - Sequence Length: 128
    - Optimizer: AdamW (lr=1e-3)

### Results

#### Training Convergence
The training loss demonstrated steady convergence, starting from ~3.34 and decreasing to ~1.03 by step 2600.

![Training Loss](results/loss_plot.png)

*Figure 1: Training and Validation Loss over steps.*

#### Evaluation Metrics
Evaluated on the Wikitext-2 validation set:

- **Average Validation Loss:** 1.2365
- **Perplexity (Byte-level):** 3.4434

This perplexity indicates the model has successfully learned structural patterns in the byte sequence, significantly outperforming random chance (Perplexity ~256).

#### Weight Analysis
The "Dragon Hatchling" hypothesis suggests that trained weights significantly deviate from their initialization, potentially exhibiting heavy-tailed distributions similar to biological synapses.

![Weight Distribution](results/weight_distribution.png)

*Figure 2: Weight distributions for learned parameters.*

### Conclusion
The BDH-GPU implementation is functional and capable of learning from text data. The integration with Hugging Face Hub allows for easy sharing and remote experimentation. Further work is needed to scale the model (larger $N$, more layers) and evaluate on larger datasets to fully verify the scaling laws proposed in the paper.

## License

MIT
