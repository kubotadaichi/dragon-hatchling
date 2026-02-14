# Dragon Hatchling (BDH) Implementation Report

## Overview
This report documents the implementation and preliminary experimental results of the Dragon Hatchling (BDH) model, a biologically inspired neural network architecture that bridges the gap between Transformers and brain models.

**Repository:** `https://github.com/kubotadaichi/dragon-hatchling`
**Model on HF Hub:** `daichi202/dragon-hatchling`

## Implementation Details
We implemented the GPU-friendly version of the model, **BDH-GPU**, as described in Appendix E of the paper ["Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain"](https://arxiv.org/abs/2509.26507).

### Key Components
- **BDH-GPU Architecture:** Implemented in `model.py`.
- **Linear Attention:** Implemented with `O(TK)` complexity.
- **Rotary Positional Embeddings (RoPE):** Added to enhance positional awareness (a deviation/addition to the original provided snippet).
- **Training Script:** `experiment.py` handles data loading (Wikitext-2), training loop, and Hugging Face Hub integration.

## Experimental Setup
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

## Results

### Training Convergence
The training loss demonstrated steady convergence, starting from ~3.34 and decreasing to ~1.03 by step 2600.

![Training Loss](results/loss_plot.png)

*Figure 1: Training and Validation Loss over steps.*

### Evaluation Metrics
Evaluated on the Wikitext-2 validation set:

- **Average Validation Loss:** 1.2365
- **Perplexity (Byte-level):** 3.4434

This perplexity indicates the model has successfully learned structural patterns in the byte sequence, significantly outperforming random chance (Perplexity ~256).

### Weight Analysis
The "Dragon Hatchling" hypothesis suggests that trained weights significantly deviate from their initialization, potentially exhibiting heavy-tailed distributions similar to biological synapses.

![Weight Distribution](results/weight_distribution.png)

*Figure 2: Weight distributions for learned parameters.*

## Conclusion
The BDH-GPU implementation is functional and capable of learning from text data. The integration with Hugging Face Hub allows for easy sharing and remote experimentation. Further work is needed to scale the model (larger $N$, more layers) and evaluate on larger datasets to fully verify the scaling laws proposed in the paper.
