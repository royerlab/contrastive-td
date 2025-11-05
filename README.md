# Contrastive TD

A Python library for applying contrastive learning to multi-object tracking data.

## Overview

Contrastive TD implements auxiliary routines to implement contrastive learning on multi-object tracking data.

It builds on [tracksdata](https://github.com/royerlab/tracksdata), a standardized framework for representing tracking problems as graphs, where detections are nodes and connections are edges.

This library provides:

- **TripletDataset**: Converts tracking graphs into triplet datasets (anchor, positive, negative) for contrastive learning
- **Training utilities**: A flexible training loop supporting custom loss functions, optimizers, and learning rate schedulers
- **Graph-based sampling**: Automatically generates triplets from ground truth tracking annotations

## Use Cases

- Learning discriminative embeddings for object re-identification in tracking scenarios
- Training neural networks to distinguish between true and false object associations
- Improving tracking performance through learned similarity metrics

## Installation

```bash
pip install git+https://github.com/royerlab/contrastive-td.git
```

## Quick Start

```python
import torch
import tracksdata as td
from contrastive_td.data import TripletDataset
from contrastive_td.fitting import training_loop

# Load your tracking graph
graph = td.load_graph("path/to/tracking/data")

# Create triplet dataset
dataset = TripletDataset(
    graph=graph,
    node_feature_key="features",
    edge_ground_truth_key="is_true_link"
)

# Define your embedding model
model = torch.nn.Sequential(
    torch.nn.Linear(input_dim, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64)
)

# Train with your loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_loop(
    dataset=dataset,
    model=model,
    epochs=10,
    loss_func=your_triplet_loss,
    opt=optimizer
)
```

## License

MIT
