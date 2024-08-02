import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dgl
import gin
import pandas as pd
import torch
from featurizers import ReactionFeaturizer
from torch.utils.data import Dataset
from torchtyping import TensorType
from tqdm import tqdm


@gin.configurable()
class ReactionDataset(Dataset):
    """
    A dataset of reactions.
    """

    def __init__(
        self, split_path: str, preprocessed_dir: str, sep: str = None, contrastive: bool = False
    ):
        print(split_path)
        print(preprocessed_dir)
        self.df = pd.read_csv(split_path, sep=sep)
        self.split_name = Path(split_path).stem
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir is not None else None
        self.smiles_to_path: Optional[Dict[str, str]] = None
        if self.preprocessed_dir is not None:
            path = self.preprocessed_dir / "smiles_to_path.json"
            if path.exists():
                with open(path, "r") as fp:
                    self.smiles_to_path = json.load(fp)
        self.reactants_list: List[List[str, ...]] = [
            r.split(".") for r in self.df["reactants"].tolist()
        ]
        self.product_list: List[str] = self.df["product"].tolist()
        self.labels = self.df["feasible"].tolist() if "feasible" in self.df else None

        if not self.is_preprocessed():
            self.featurizer = ReactionFeaturizer()
        else:
            new_smiles_to_path = {}
            for product in self.product_list:
                new_smiles_to_path[product] = self.smiles_to_path[product]
            for reactants in self.reactants_list:
                for reactant in reactants:
                    new_smiles_to_path[reactant] = self.smiles_to_path[reactant]
            self.smiles_to_path = new_smiles_to_path

        self.contrastive = contrastive
        if self.contrastive:
            print(sum(self.labels), len(self.labels))
            assert len(self.labels) % sum(self.labels) == 0
            self.group_k_consecutive = len(self.labels) // sum(self.labels)
        else:
            self.group_k_consecutive = 1

    def __len__(self):
        return len(self.df) // self.group_k_consecutive

    def is_preprocessed(self):
        if self.smiles_to_path is None:
            path = self.preprocessed_dir / "smiles_to_path.json"
            if path.exists():
                with open(path, "r") as fp:
                    self.smiles_to_path = json.load(fp)
        return self.smiles_to_path is not None

    def load_graph(self, smiles: str):
        path = self.preprocessed_dir / self.smiles_to_path[smiles]
        return dgl.load_graphs(str(path))[0][0]

    def featurize_graph(self, smiles: str):
        return self.featurizer.featurize_smiles_single(smiles)

    def _get_underling_item(self, index: int) -> Tuple[List[dgl.DGLGraph], dgl.DGLGraph, float]:
        reactants = self.reactants_list[index]
        product = self.product_list[index]
        label = self.labels[index] if self.labels is not None else 0.0
        if self.is_preprocessed():
            reactant_graphs = [self.load_graph(r) for r in reactants]
            product_graph = self.load_graph(product)
        else:
            reactant_graphs = [self.featurize_graph(r) for r in reactants]
            product_graph = self.featurize_graph(product)
        return reactant_graphs, product_graph, label

    def __getitem__(self, index: int) -> List[Tuple[List[dgl.DGLGraph], dgl.DGLGraph, float]]:
        items = []
        for i in range(index * self.group_k_consecutive, (index + 1) * self.group_k_consecutive):
            items.append(self._get_underling_item(i))
        return items

    def preprocess(self):
        if self.is_preprocessed() or self.preprocessed_dir is None:
            return
        smiles_to_path = {}
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(self.df)), desc=f"Preprocessing {self.split_name}"):
            reactant_graphs, product_graph, _ = self._get_underling_item(i)
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
        self.smiles_to_path = smiles_to_path
        self.featurizer = None

    def collate(
        self, items: List[List[Tuple[List[dgl.DGLGraph], dgl.DGLGraph, float]]]
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, TensorType[float]]:
        reactant_graphs_list, product_graph_list, labels_list = [], [], []
        for item_list in items:
            for reactant_graphs, product_graph, label in item_list:
                reactant_graphs_list.append(dgl.merge(reactant_graphs))
                product_graph_list.append(product_graph)
                labels_list.append(label)

        return (
            dgl.batch(reactant_graphs_list),
            dgl.batch(product_graph_list),
            torch.tensor(labels_list).float(),
        )
