from typing import Protocol
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from contrastive_td.data import TripletDataset


class LossFunc(Protocol):
    def __call__(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        anchor : torch.Tensor
            The anchor embedding.
        positive : torch.Tensor
            The positive embedding.
        negatives : torch.Tensor
            The negative embeddings.
        Returns the loss for a batch of triplets.
        """
        raise NotImplementedError


def training_loop(
    dataset: TripletDataset,
    model: torch.nn.Module,
    epochs: int,
    loss_func: LossFunc,
    opt: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """
    Train a model on a dataset.

    Parameters
    ----------
    dataset : TripletDataset
        The dataset to train on.
    model : torch.nn.Module
        D-dimensional to K-dimensional embedding model.
    epochs : int
        The number of epochs to train for.
    loss_func : LossFunc
        The loss function to use.
    opt : torch.optim.Optimizer
        The optimizer to use.
    lr_scheduler : torch.optim.lr_scheduler.LRScheduler | None, optional
        The learning rate scheduler to use, applied after each epoch.
    """

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_steps = 0
        
        pbar = tqdm(dataset, desc=f"Epoch {epoch}", total=len(dataset))

        for anchor, positive, negatives in pbar:
            opt.zero_grad()
            anchor = model(anchor)
            positive = model(positive)
            negatives = model(negatives)
            loss = loss_func(anchor, positive, negatives)
            loss.backward()
            opt.step()

            with torch.no_grad():
                loss = loss.item()
                pbar.set_postfix(loss=loss)
                total_loss += loss
                total_steps += 1
        
        print(f"Epoch {epoch} loss: {total_loss / total_steps}")

        if lr_scheduler is not None:
            lr_scheduler.step()
