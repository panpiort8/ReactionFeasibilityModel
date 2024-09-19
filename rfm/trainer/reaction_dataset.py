import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dgl
import gin
import pandas as pd
import torch
from featurizers import ReactionFeaturizer
from torch import Tensor
from torch.utils.data import Dataset
from torchtyping import TensorType
from tqdm import tqdm


@gin.configurable()
class ReactionDataset(Dataset):
    """
    A dataset of reactions.
    """

    def __init__(self, split_path: str, preprocessed_dir: str | None, sep: str = None):
        self.df = pd.read_csv(split_path, sep=sep)
        self.split_name = Path(split_path).stem
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir is not None else None
        self.smiles_to_path: Optional[Dict[str, str]] = None
        self.reactants_list: List[List[str, ...]] = [
            r.split(".") for r in self.df["reactants"].tolist()
        ]
        self.product_list: List[str] = self.df["product"].tolist()
        self.labels = self.df["feasible"].tolist()
        self.featurizer = None
        if self.preprocessed_dir is not None:
            smiles_path = self.preprocessed_dir / "smiles_to_path.json"
            if smiles_path.exists():
                with open(smiles_path, "r") as fp:
                    self.smiles_to_path = json.load(fp)

        self.preprocess()

    def __len__(self):
        return len(self.df)

    def is_preprocessed(self):
        return self.smiles_to_path is not None

    def load_graph(self, smiles: str):
        path = self.preprocessed_dir / self.smiles_to_path[smiles]
        return dgl.load_graphs(str(path))[0][0]

    def featurize_graph(self, smiles: str):
        return self.featurizer.featurize_smiles_single(smiles)

    def __getitem__(self, index: int) -> Tuple[List[dgl.DGLGraph], dgl.DGLGraph, float]:
        reactants = self.reactants_list[index]
        product = self.product_list[index]
        label = self.labels[index]
        if self.is_preprocessed():
            reactant_graphs = [self.load_graph(r) for r in reactants]
            product_graph = self.load_graph(product)
        else:
            reactant_graphs = [self.featurize_graph(r) for r in reactants]
            product_graph = self.featurize_graph(product)
        return reactant_graphs, product_graph, label

    def preprocess(self):
        if self.is_preprocessed() or self.preprocessed_dir is None:
            return
        self.featurizer = ReactionFeaturizer()
        smiles_to_path = {}
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(self.df)), desc=f"Preprocessing {self.split_name}"):
            reactant_graphs, product_graph, _ = self[i]
            reactants_smiles = self.reactants_list[i]
            product_smiles = self.product_list[i]
            all_graphs = reactant_graphs + [product_graph]
            all_smiles = reactants_smiles + [product_smiles]
            for smiles, graph in zip(all_smiles, all_graphs):
                if smiles not in smiles_to_path:
                    smiles_pickle_path = f"{len(smiles_to_path)}.pkl"
                    smiles_to_path[smiles] = smiles_pickle_path
                    dgl.save_graphs(str(self.preprocessed_dir / smiles_pickle_path), [graph])
        with open(self.preprocessed_dir / "smiles_to_path.json", "w") as fp:
            json.dump(smiles_to_path, fp)
        self.featurizer = None
        self.smiles_to_path = smiles_to_path

    def collate(
        self, items: List[Tuple[List[dgl.DGLGraph], dgl.DGLGraph, float]]
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, Tensor]:
        reactant_graphs_list, product_graph_list, labels_list = [], [], []

        for reactant_graphs, product_graph, label in items:
            reactant_graphs_list.append(dgl.merge(reactant_graphs))
            product_graph_list.append(product_graph)
            labels_list.append(label)

        return (
            dgl.batch(reactant_graphs_list),
            dgl.batch(product_graph_list),
            torch.tensor(labels_list).float(),
        )
