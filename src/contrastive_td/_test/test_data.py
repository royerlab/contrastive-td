import pytest
import numpy as np
import torch
import tracksdata as td

from contrastive_td.data import TripletDataset


@pytest.fixture
def mock_graph() -> td.graph.InMemoryGraph:

    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("x", 0.0)
    graph.add_node_attr_key("feat",  np.zeros(3))
    graph.add_edge_attr_key("ground_truth", False)

    n0 = graph.add_node({td.DEFAULT_ATTR_KEYS.T: 0, "x": 0, "feat": np.array([0, 0, 0])})
    n1 = graph.add_node({td.DEFAULT_ATTR_KEYS.T: 0, "x": 1, "feat": np.array([1, 1, 1])})
    n2 = graph.add_node({td.DEFAULT_ATTR_KEYS.T: 0, "x": 2, "feat": np.array([2, 2, 2])})

    n3 = graph.add_node({td.DEFAULT_ATTR_KEYS.T: 1, "x": 3, "feat": np.array([3, 3, 3])})
    n4 = graph.add_node({td.DEFAULT_ATTR_KEYS.T: 1, "x": 4, "feat": np.array([4, 4, 4])})
    n5 = graph.add_node({td.DEFAULT_ATTR_KEYS.T: 1, "x": 5, "feat": np.array([5, 5, 5])})

    graph.add_edge(n0, n3, {"ground_truth": True})
    graph.add_edge(n0, n4, {"ground_truth": False})
    graph.add_edge(n0, n5, {"ground_truth": False})

    graph.add_edge(n1, n3, {"ground_truth": False})
    graph.add_edge(n1, n4, {"ground_truth": True})
    graph.add_edge(n1, n5, {"ground_truth": True})

    graph.add_edge(n2, n3, {"ground_truth": False})
    graph.add_edge(n2, n4, {"ground_truth": False})
    graph.add_edge(n2, n5, {"ground_truth": False})

    return graph


def test_triplet_dataset(mock_graph: td.graph.InMemoryGraph) -> None:
    dataset = TripletDataset(mock_graph, "feat", "ground_truth")
    assert len(dataset) == 3

    first_triplet = dataset[0]
    torch.testing.assert_close(first_triplet[0], torch.Tensor([3, 3, 3]))
    torch.testing.assert_close(first_triplet[1], torch.Tensor([0, 0, 0]))
    torch.testing.assert_close(first_triplet[2], torch.Tensor([[1, 1, 1], [2, 2, 2]]))

    second_triplet = dataset[1]
    torch.testing.assert_close(second_triplet[0], torch.Tensor([4, 4, 4]))
    torch.testing.assert_close(second_triplet[1], torch.Tensor([1, 1, 1]))
    torch.testing.assert_close(second_triplet[2], torch.Tensor([[0, 0, 0], [2, 2, 2]]))

    third_triplet = dataset[2]
    torch.testing.assert_close(third_triplet[0], torch.Tensor([5, 5, 5]))
    torch.testing.assert_close(third_triplet[1], torch.Tensor([1, 1, 1]))
    torch.testing.assert_close(third_triplet[2], torch.Tensor([[0, 0, 0], [2, 2, 2]]))
