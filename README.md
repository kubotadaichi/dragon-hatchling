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

## Results

See [report.md](report.md) for detailed experimental results.

**Validation Perplexity:** ~3.44 (Byte-level)

## License

MIT
