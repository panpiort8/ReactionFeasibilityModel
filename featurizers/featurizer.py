from typing import List, Tuple

import dgl
from dgl import DGLGraph
from dgllife.utils import CanonicalBondFeaturizer, mol_to_bigraph, WeaveAtomFeaturizer
from rdkit import Chem

from featurizers.utils import ATOM_TYPES


class ReactionFeaturizer:
    def __init__(self):
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=ATOM_TYPES)

    def featurize_smiles_single(self, smiles: str) -> DGLGraph:
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveHs(mol)
        return mol_to_bigraph(
            mol=mol,
            add_self_loop=True,
            canonical_atom_order=False,
            node_featurizer=self.node_featurizer,
            edge_featurizer=self.edge_featurizer,
        )

    def featurize_reaction_single(self, reactants: List[str], product: str) -> Tuple[DGLGraph, DGLGraph]:
        reactants_graphs = dgl.merge([self.featurize_smiles_single(r) for r in reactants])
        product_graph = self.featurize_smiles_single(product)
        return reactants_graphs, product_graph

    def featurize_reactions_batch(self, reaction_list: List[Tuple[List[str], str]]) -> Tuple[DGLGraph, DGLGraph]:
        reactions = []
        for reactants, product in reaction_list:
            reaction = self.featurize_reaction_single(reactants, product)
            reactions.append(reaction)
        return self.collate_reactions(reactions)

    def collate_reactions(self, reactions: List[Tuple[DGLGraph, DGLGraph]]) -> Tuple[DGLGraph, DGLGraph]:
        reactants, products = zip(*reactions)
        reactants_batch = dgl.batch(reactants)
        product_batch = dgl.batch(products)
        return reactants_batch, product_batch
