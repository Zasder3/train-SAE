# train-SAE
Train Sparse Autoencoders (SAEs) using PyTorch.

This repository provides a flexible framework for training sparse autoencoders on various model activations, with a primary focus on transformer models for arithmetic tasks. While the framework supports protein language models (ESM2), this functionality has not been experimentally explored yet.

## Installation

```language=bash
# Clone the repository
git clone https://github.com/Zasder3/train-SAE.git
cd train-SAE

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

To train a sparse autoencoder:

```language=bash
# Train on arithmetic tasks (grokking)
python -m train_sae.train.main --config_file configs/train-crosscoder.json
```


## Configuration Basics

The framework uses a configuration system based on Pydantic models. You can configure your training run in two ways:

1. Using a JSON configuration file (recommended)
2. Passing command-line arguments

Example configuration file:

```language=json
{
    "dataset_dir": "path/to/dataset",
    "task": "grokking",
    "task_kwargs": {
        "prime": 97
    },
    "num_steps": 10000,
    "batch_size": 64,
    "lr": 1e-3,
    "featurizing_model_name": "path/to/grokking-model",
    "n_layers": 1,
    "sparsity": 5.0,
    "sparse_dim": 2048,
    "featurizing_model_type": "transformer",
    "featurizing_model_kwargs": {
        "vocab_size": 99,
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 4,
        "use_geglu": false
    }
}
```


Command-line override example:

```language=bash
python -m train_sae.train.main --config_file configs/train-crosscoder.json --lr 1e-4 --num_steps 20000
```

## Training on Arithmetic Tasks (Grokking)

The framework supports training on various arithmetic tasks, which can help study the phenomenon of "grokking" - where models suddenly generalize after extended training.

Available arithmetic tasks:
- Addition
- Subtraction
- Multiplication
- Division
- Conditional operations


To train on different operations, simply change the `task_name` parameter:

```language=bash
### For multiplication
python -m train_sae.train.main --config_file configs/grokking-base.json --task_kwargs.task_name multiplication

### For division
python -m train_sae.train.main --config_file configs/grokking-base.json --task_kwargs.task_name division
```

## Model Types

The framework supports two types of models:

1. **Transformers**: Custom transformer implementations for tasks like grokking (primary focus)
2. **ESM2**: Protein language models from Meta AI (supported but not experimentally explored)

You can configure which model to use with the `featurizing_model_type` parameter.

## SAE Types

Two types of sparse autoencoders are available:

1. **Vanilla SAE**: Uses ReLU activation and L1 regularization for sparsity
2. **TopK SAE**: Only keeps the top-k activations per feature dimension

You can select the SAE type with the `sae_type` parameter:

```language=bash
python -m train_sae.train.main --config_file configs/train.json --sae_type topk --sae_kwargs.topk 50
```


## Cross-Coder SAEs

The framework supports training a single SAE on multiple model layers or even across different models using the CrossCoderSAE feature:

```language=json
{
    "featurizing_model_name": ["model1", "model2"],
    "n_layers": [8, 16, 24],
    "sparse_dim": 4096
}
```

This configuration will train an SAE on activations from layers 8, 16, and 24 of both model1 and model2.

## Experiment Tracking

The framework uses Weights & Biases for experiment tracking. To configure:

```language=json
{
    "project_name": "my-sae-experiments",
    "run_name": "experiment-1"
}
```


Key metrics tracked include:
- Reconstruction loss
- Sparsity metrics (L0 norm)
- Dead neuron count
- Language modeling performance (for LM tasks)

## Available Datasets

For grokking tasks, the framework generates the arithmetic problem datasets automatically. For protein language models, FASTA format datasets can be used, but this has not been experimentally explored.

## Advanced Configuration Options

| Parameter | Description |
|-----------|-------------|
| `device` | Device to train on (`cpu`, `cuda`, `cuda:0`, etc.) |
| `dtype` | Data type to use (`float32`, `float16`, etc.) |
| `compile` | Whether to use PyTorch compilation for speed |
| `sparsity_warmup_steps` | Steps to warm up the sparsity penalty |
| `lr_scheduler` | Learning rate scheduler (`constant`, `cosine`, `linear_decay`) |
| `lr_warmup_steps` | Number of warmup steps for the learning rate |

## Example Experiments

Here are some example experiments you can try:

1. **Compare different SAE dimensions**:
   ```bash
   python -m train_sae.train.main --config_file configs/train.json --sparse_dim 1024
   python -m train_sae.train.main --config_file configs/train.json --sparse_dim 2048
   python -m train_sae.train.main --config_file configs/train.json --sparse_dim 4096
   ```

2. **Compare vanilla vs. topk SAEs**:
   ```bash
   python -m train_sae.train.main --config_file configs/train.json --sae_type vanilla
   python -m train_sae.train.main --config_file configs/train.json --sae_type topk --sae_kwargs.topk 50
   ```

3. **Study grokking with different learning rates**:
   ```bash
   python -m train_sae.train.main --config_file configs/train-crosscoder.json --lr 1e-2
   python -m train_sae.train.main --config_file configs/train-crosscoder.json --lr 1e-3
   python -m train_sae.train.main --config_file configs/train-crosscoder.json --lr 1e-4
   ```

## Core Module Structure

```language=
train_sae/
├── configs/        # Configuration utilities
│   ├── base.py     # Base configuration classes
│   └── utils.py    # Utility functions for configurations
├── models/         # Model implementations
│   ├── esm2.py     # ESM2 model implementation
│   ├── model.py    # Abstract model interfaces
│   └── transformer.py # Transformer model implementation
├── saes/           # Sparse autoencoder implementations
│   ├── crosscoder.py # Cross-coder SAE implementation
│   ├── topk.py     # TopK SAE implementation 
│   └── vanilla.py  # Vanilla SAE implementation
└── train/          # Training utilities
    ├── datasets/   # Dataset implementations
    │   └── fasta.py # FASTA dataset implementation
    ├── main.py     # Main training script
    ├── scheduler.py # Learning rate schedulers
    ├── tasks.py    # Task implementations
    └── train.py    # Training loop implementation
```

## Experimental Directory Walkthrough

The `experimental/` directory contains Jupyter notebooks that demonstrate specific experiments and visualizations for understanding SAE behavior on arithmetic tasks. Here, we share code walking through reproducing all figures in our paper.

### Arithmetic Task Notebooks

These notebooks explore how SAEs learn to represent arithmetic operations:

- **cc_first-division.ipynb**: Investigates division-specific SAEs by comparing pre vs post-generalization feature activations and demonstrating how clamping specific neurons during training allows us to steer targeted drops in generalization performance.
- **cc_first-multiplication.ipynb**: Analyzes multiplication-specific SAEs, tracking accuracy shifts between pre and post-generalization phases and revealing how steering via activation clamping prevents the model from generalizing multiplication patterns.
- **cc_first-addition.ipynb**: Explores addition-specific SAEs, measuring performance differences before and after generalization points and testing how steering via activation clamping of critical neurons impacts the model's ability to generalize addition operations.
- **cc_first-subtraction.ipynb**: Examines subtraction-specific SAEs, comparing pre/post-generalization accuracy and identifying key neurons whose activations, when clamped, directly steer the model's ability to generalize to modular subtraction.

### Visualization Notebooks

These notebooks provide tools to interpret and visualize SAE representations:

- **viz_neuron_activations.ipynb**: Visualizes individual neuron activations across different inputs, helping identify what specific neurons have learned.
- **viz_feature_attribution.ipynb**: Maps SAE features back to input tokens, showing which parts of the input most strongly activate specific neurons.
- **viz_latent_space.ipynb**: Provides dimensionality reduction visualizations (t-SNE, UMAP) of the SAE's latent space to reveal clustering of similar operations.
- **viz_neuron_interpretability.ipynb**: Offers interpretability techniques to understand what mathematical concepts individual neurons represent.

These experimental notebooks serve as both examples of how to use the framework for specific research questions and as demonstrations of SAE behavior on arithmetic tasks. They can be adapted for your own investigations or used as templates for new experiments.

## License

[MIT License](LICENSE)

## Citation

If you use this framework in your research, please cite:

```language=bibtex
@software{train_sae,
  title = {train-SAE: Training Sparse Autoencoders using PyTorch},
  year = {2025},
  url = {https://github.com/yourusername/train-SAE}
}
```


## Contributing
Contributions are welcome! Please feel free to submit a pull request, or raise an issue for new feature requests.