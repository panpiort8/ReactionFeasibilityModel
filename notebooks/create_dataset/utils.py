from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles


def remove_atom_mapping(smiles: str) -> str:
    mol = MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")
    return MolToSmiles(mol)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return MolToSmiles(mol)


def get_ground_truth_dict(data_dir: Path, split: str):
    positive_df = pd.read_csv(data_dir / "positive" / f"{split}.csv")
    product_to_reactants = defaultdict(set)
    for _, row in positive_df.iterrows():
        product_to_reactants[row["product"]].add(row["reactants"])
    if split != "train":
        train_product_to_reactants = get_ground_truth_dict("train")
        for product, reactants in train_product_to_reactants.items():
            product_to_reactants[product] |= reactants
    return product_to_reactants
