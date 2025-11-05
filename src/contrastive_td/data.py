from typing import NamedTuple

import torch
import tracksdata as td
import polars as pl
from torch.utils.data import Dataset


class Triplet(NamedTuple):
    anchor_id: int
    positive_id: int
    negative_ids: torch.Tensor


class TripletDataset(Dataset):
    """
    A dataset of triplets.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to use.
    node_feature_key : str
        The key of the node feature to use, should be a D-dimensional feature vector per node.
    edge_ground_truth_key : str
        The key of the edge ground truth to use, should be a boolean vector per edge.

    Attributes
    ----------
    _graph : td.graph.BaseGraph
        The graph to use.
    _node_features : dict[int, torch.Tensor]
        A dictionary of node features.
    _triplets : list[Triplet]
        A list of triplets.
    """
    def __init__(
        self,
        graph: td.graph.BaseGraph,
        node_feature_key: str,
        edge_ground_truth_key: str,
    ):
        self._graph = graph
        node_attrs: pl.DataFrame = graph.node_attrs([td.DEFAULT_ATTR_KEYS.NODE_ID, node_feature_key])

        self._node_features = {
            node_id: feat
            for node_id, feat in zip(
                node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
                node_attrs[node_feature_key].to_torch().float(),
            )
        }

        edge_attrs: pl.DataFrame = graph.edge_attrs()

        self._triplets = []

        for (anchor_id,), group in edge_attrs.group_by(td.DEFAULT_ATTR_KEYS.EDGE_TARGET):
            mask = group[edge_ground_truth_key]
            source_ids = group[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_torch()
            positive_ids = source_ids[mask]
            negative_ids = source_ids[~mask]

            for positive_id in positive_ids:
                self._triplets.append(
                    Triplet(anchor_id, positive_id, negative_ids)
                )

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        index : int
            The index of the triplet to get.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - anchor : (1, D)-dimensional anchor feature tensor
            - positive : (1, D)-dimensional positive feature tensor
            - negative : (M, D)-dimensional negative feature tensor, where M is the number of negative samples
        """
        triplet = self._triplets[index]
        return tuple(
            self._node_features[node_id]
            for node_id in triplet
        )
