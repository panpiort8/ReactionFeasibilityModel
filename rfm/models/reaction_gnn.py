from typing import Literal

import dgl
import gin
import torch
from torch import nn
from torchtyping import TensorType

from rfm.models.gnns import AttentionGNN


@gin.configurable()
class ReactionGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        edge_in_dim: int = 13,
        node_in_dim: int = 80,
        use_attention: bool = True,
        concat_type: Literal["simple", "fancy"] = "simple",
        mlp_dropout: float = 0.4,
        attention_dropout: float = 0.1,
        gnn_type: str = 'mpnn',
        checkpoint_path: str | None = None
    ):
        super().__init__()
        assert concat_type in ["simple", "fancy"]
        self.dropout = mlp_dropout
        self.reactants_gnn = AttentionGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            edge_in_dim=edge_in_dim,
            node_in_dim=node_in_dim,
            use_attention=use_attention,
            attention_dropout=attention_dropout,
            gnn_type=gnn_type
        )
        self.product_gnn = AttentionGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            edge_in_dim=edge_in_dim,
            node_in_dim=node_in_dim,
            use_attention=use_attention,
            attention_dropout=attention_dropout,
            gnn_type=gnn_type
        )
        self.concat_type = concat_type
        mlp_size = hidden_dim * 2 if concat_type == "simple" else hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_size, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, 1),
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            self.load_state_dict(state_dict)

    def concat(
        self, reactants_embeddings: TensorType[float], products_embeddings: TensorType[float]
    ) -> TensorType[float]:
        if self.concat_type == "simple":
            x = torch.cat([reactants_embeddings, products_embeddings], dim=-1)
        else:
            x = torch.cat(
                [
                    reactants_embeddings,
                    reactants_embeddings - products_embeddings,
                    reactants_embeddings * products_embeddings,
                    products_embeddings,
                ],
                dim=-1,
            )
        return x

    def forward(self, reactants: dgl.DGLGraph, products: dgl.DGLGraph) -> TensorType[float]:
        reactants_embeddings = self.reactants_gnn(reactants)
        products_embeddings = self.product_gnn(products)
        embeddings = self.concat(reactants_embeddings, products_embeddings)
        feasibility = self.mlp(embeddings).squeeze(-1)
        return feasibility





