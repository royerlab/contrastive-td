# Contrastive TD

A toolbox for contrastive learning on [tracksdata](https://github.com/royerlab/tracksdata) graphs.

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
