from typing import Iterable, Tuple

import torch
from torch_geometric.utils import to_dense_batch
from torchtyping import TensorType


def to_indices(counts: TensorType[int]) -> TensorType[int]:
    indices = torch.arange(len(counts), device=counts.device)
    return torch.repeat_interleave(indices, counts).long()


def to_dense_embeddings(
        embeddings: torch.Tensor,
        counts: Iterable[int],
        fill_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts sparse node embeddings to dense node embeddings with padding.
    Arguments:
        embeddings: embeddings in a sparse format, i.e. [total_num_nodes, hidden_size]
        graph: a batch of graphs
        fill_value: a value to fill the padding with
    Returns:
        node_embeddings: embeddings in a dense format, i.e. [batch_size, max_num_nodes or max_num_edges, hidden_size]
        mask: a mask indicating which nodes are real and which are padding, i.e. [batch_size, max_num_nodes]
    """
    counts = (
        torch.tensor(counts, device=embeddings.device)
        if not isinstance(counts, torch.Tensor)
        else counts
    )
    indices = torch.arange(len(counts), device=counts.device)
    batch = torch.repeat_interleave(indices, counts).long()  # e.g. [0, 0, 1, 1, 1, 2, 2, 2]
    return to_dense_batch(
        embeddings, batch, fill_value=fill_value
    )  # that's the only reason we have torch_geometric in the requirements
