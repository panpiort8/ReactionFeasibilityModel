import math
from abc import abstractmethod, ABC
from typing import Any, Type, Dict

import dgl
import torch
from dgllife.model import MPNNGNN, GAT
from torch import nn
from torch_geometric.utils import to_dense_batch
from torchtyping import TensorType

from models.utils import to_indices


class AttentionGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        edge_in_dim: int,
        node_in_dim: int,
        use_attention: bool,
        attention_dropout: float,
        gnn_type: str
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.edge_in_dim = edge_in_dim
        self.node_in_dim = node_in_dim
        self.use_attention = use_attention
        self.gnn_type = gnn_type
        if self.gnn_type == 'mpnn':
            self.gnn = MPNNGNN(
                node_in_feats=node_in_dim,
                edge_in_feats=edge_in_dim,
                node_out_feats=hidden_dim,
                edge_hidden_feats=hidden_dim,
                num_step_message_passing=num_layers,
            )
        elif self.gnn_type == 'gat':
            self.gnn = GAT(
                in_feats=node_in_dim,
                hidden_feats=[hidden_dim // 4] * num_layers,
            )
        elif 'our' in self.gnn_type:
            gnn_name = self.gnn_type.split('_')[1]
            self.gnn = MPNNModel(
                node_features_size=node_in_dim,
                edge_features_size=edge_in_dim,
                hidden_size=hidden_dim,
                mpnn_layer_cls={
                    'gine': GINELayer,
                    'gat': GATLayer
                }[gnn_name],
                mpnn_layer_kwargs={},
                n_layers=num_layers,
                random_walk_size=0,
                residual=True,
                normalization=True,
            )

        if self.use_attention:
            self.attention = GlobalReactivityAttention(
                heads=num_attention_heads, d_model=hidden_dim, dropout=attention_dropout
            )

        self.gating_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.device = "cpu"

    def set_device(self, device: str):
        self.device = device
        super().to(device)

    def forward(self, batch: dgl.DGLGraph) -> TensorType[float]:
        x = self.gnn(batch, batch.ndata["h"], batch.edata["e"])  # (total_num_nodes, hidden_dim
        graph_indices = to_indices(batch.batch_num_nodes())
        x, mask = to_dense_batch(x=x, batch=graph_indices)
        if self.use_attention:
            _, x = self.attention.forward(x, mask=mask)  # (batch_size, max_num_nodes, hidden_dim)
            x = torch.masked_fill(x, ~mask.unsqueeze(-1), 0.0)
        gating_scores = self.gating_mlp(x).squeeze(-1)  # (batch_size, max_num_nodes)
        gating_scores = torch.masked_fill(gating_scores, ~mask, -float("inf"))
        gating_scores = torch.softmax(gating_scores, dim=1)
        x = torch.sum(x * gating_scores.unsqueeze(-1), dim=1)
        return x

class MPNNLayerBase(ABC, nn.Module):
    def _init(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    @abstractmethod
    def forward(
            self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, graph: dgl.DGLGraph
    ) -> torch.Tensor:
        ...


class MPNNModel(nn.Module):
    def __init__(
            self,
            node_features_size: int,
            edge_features_size: int,
            hidden_size: int,
            mpnn_layer_cls: Type[MPNNLayerBase],
            mpnn_layer_kwargs: Dict[str, Any],
            n_layers: int,
            normalization: bool,
            residual: bool,
            random_walk_size: int,
    ):
        super().__init__()
        self.random_walk_size = random_walk_size
        self.linear_node = nn.Linear(node_features_size + random_walk_size, hidden_size)
        self.linear_edge = nn.Linear(edge_features_size, hidden_size)
        self.mpnn_layers = nn.ModuleList(
            [mpnn_layer_cls(hidden_size=hidden_size, **mpnn_layer_kwargs) for _ in range(n_layers)]
        )
        if normalization:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        else:
            self.layer_norms = None
        self.residual = residual
        self.n_layers = n_layers

    def forward(self, graph: dgl.DGLGraph, *args, **kwargs) -> torch.Tensor:
        node_embeddings, edge_embeddings = graph.ndata["h"], graph.edata["e"]
        if self.random_walk_size > 0:
            random_walk_pe = dgl.random_walk_pe(graph, k=self.random_walk_size)
            node_embeddings = torch.cat([node_embeddings, random_walk_pe], dim=-1)
        node_embeddings = self.linear_node(node_embeddings)
        edge_embeddings = self.linear_edge(edge_embeddings)
        for i in range(self.n_layers):
            new_node_embeddings = self.mpnn_layers[i](
                node_embeddings=node_embeddings, edge_embeddings=edge_embeddings, graph=graph
            )
            if self.residual:
                node_embeddings = new_node_embeddings + node_embeddings
            else:
                node_embeddings = new_node_embeddings
            if self.layer_norms:
                node_embeddings = self.layer_norms[i](node_embeddings)
        return node_embeddings


class GINELayer(MPNNLayerBase):
    def __init__(self, hidden_size: int, eps: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def forward(
            self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, graph: dgl.DGLGraph
    ) -> torch.Tensor:
        start_nodes, end_nodes, edge_ids = graph.edges(order="srcdst", form="all")
        messages = self.relu(node_embeddings[end_nodes] + edge_embeddings[edge_ids])
        message_dense, _ = to_dense_batch(messages, start_nodes.long(), fill_value=0.0)
        aggregated_message = message_dense.sum(dim=1)
        node_embeddings = (1 + self.eps) * node_embeddings + aggregated_message
        node_embeddings = self.mlp(node_embeddings)
        return node_embeddings


class GATLayer(MPNNLayerBase):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_2 = nn.Linear(3 * hidden_size, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()

    def forward(
            self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor, graph: dgl.DGLGraph
    ) -> torch.Tensor:
        start_nodes, end_nodes, edge_ids = graph.edges(order="srcdst", form="all")
        messages = self.linear_1(node_embeddings)
        messages_i = messages[start_nodes]
        messages_j = messages[end_nodes]
        edges_ij = edge_embeddings[edge_ids]

        logits = self.leaky_relu(
            self.linear_2(torch.cat([messages_i, messages_j, edges_ij], dim=-1))
        ).squeeze(-1)
        logits_dense, _ = to_dense_batch(logits, start_nodes.long(), fill_value=-float("inf"))
        messages_dense, _ = to_dense_batch(messages_j, start_nodes.long(), fill_value=0.0)
        attention = torch.softmax(logits_dense, dim=-1)

        messages_dense = attention.unsqueeze(-1) * messages_dense
        node_embeddings = messages_dense.sum(dim=1)
        return node_embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores = torch.masked_fill(scores, ~mask, -float("inf"))
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)


class GlobalReactivityAttention(nn.Module):
    def __init__(self, d_model, heads, n_layers=1, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x, mask):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x
