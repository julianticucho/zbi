# zbi

Simulation-Based Inference pipeline with SNPE (Sequential Neural Posterior Estimation). Built on PyTorch and nflows, with sequential prior truncation (tSNPE). Data stored with Zarr for reproducible experiments.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from zbi.pipeline import init, simulate, train
from zbi.examples.simple import SimuladorLineal, EmbeddingNet

# Initialize experiment
init(
    run_dir="runs/my_experiment",
    x_o=x_o,
    simulator_class=SimuladorLineal,
    embedding_class=EmbeddingNet,
    embedding_kwargs=dict(dim_in=100, dim_out=4),
    prior_low=(-3.0, -3.0),
    prior_high=(3.0, 3.0),
    dim_theta=2,
    dim_x=100,
)

# Simulate and train
simulate(run_dir, round=0, n_sims=500)
train(run_dir, round=0, n_sims=500)
```

## Project Structure

```
zbi/
├── zbi/                   # Main package
│   ├── pipeline/          # High-level API (init, simulate, train, etc.)
│   ├── data/              # ZarrStore for data management
│   ├── inference/         # Posterior class
│   ├── neural_nets/       # MAF builder
│   ├── simulators/        # Base Simulator class
│   └── utils/             # Checkpointing, plotting, truncation
├── tests/                 # Test suite
├── tutorials/             # Jupyter notebooks
├── pyproject.toml         # Package config
└── requirements.txt       # Dependencies
```

## Key Features

- **Sequential prior truncation**: Iteratively narrow the prior using bounding boxes
- **Ensemble training**: Train multiple MAFs independently and monitor KL divergence
- **Proposal history**: Save proposals per round for reproducibility
- **ZarrStore**: Efficient data storage with append/clear operations

## What Needs To Be Done

- [ ] **Update `requirements.txt`**: Dependencies may be outdated or incomplete
- [ ] **Update `tutorials/`**: Notebooks need to reflect current API changes (renamed functions, new features)
- [ ] **Add docstrings**: Most functions lack documentation
- [ ] **CI/CD**: Set up GitHub Actions for automated testing
