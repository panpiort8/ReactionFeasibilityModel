import dgl
import pytest
import torch
from dgllife.utils import CanonicalBondFeaturizer, WeaveAtomFeaturizer, mol_to_bigraph
from rdkit import Chem

from rfm.featurizers.utils import ATOM_TYPES, disjoint_union_plain


def build_graph(smiles: str) -> dgl.DGLGraph:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    Chem.RemoveHs(mol)
    node_featurizer = WeaveAtomFeaturizer(atom_types=ATOM_TYPES)
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    g = mol_to_bigraph(
        mol=mol,
        add_self_loop=True,
        canonical_atom_order=False,
        node_featurizer=node_featurizer,
        edge_featurizer=edge_featurizer,
    )
    return g


def atom_count(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    Chem.RemoveHs(m)
    return m.GetNumAtoms()


def test_disjoint_union_two_reactants_node_count_matches_sum():
    reactants = ["CCO", "N"]  # ethanol (C-C-O) and nitrogen
    graphs = [build_graph(s) for s in reactants]
    union = disjoint_union_plain(graphs)

    # Sum num_nodes from graphs
    expected_nodes = sum(g.num_nodes() for g in graphs)
    assert union.num_nodes() == expected_nodes

    # Also check equals sum of atom counts from RDKit
    expected_atoms = sum(atom_count(s) for s in reactants)
    assert expected_nodes == expected_atoms


def test_disjoint_union_three_reactants_node_count_matches_sum():
    reactants = ["C", "O", "CC"]
    graphs = [build_graph(s) for s in reactants]
    union = disjoint_union_plain(graphs)

    expected_nodes = sum(g.num_nodes() for g in graphs)
    assert union.num_nodes() == expected_nodes

    expected_atoms = sum(atom_count(s) for s in reactants)
    assert expected_nodes == expected_atoms


def test_feature_shapes_concatenate_correctly():
    reactants = ["CC", "NN"]
    graphs = [build_graph(s) for s in reactants]
    # record a representative node feature shape (excluding first dim)
    key = next(iter(graphs[0].ndata.keys()))
    per_shapes = [g.ndata[key].shape for g in graphs]
    feat_dim = per_shapes[0][1:]

    union = disjoint_union_plain(graphs)
    assert key in union.ndata
    # First dimension should equal sum of first dims
    assert union.ndata[key].shape[0] == sum(s[0] for s in per_shapes)
    # Remaining dims equal original feature dims
    assert union.ndata[key].shape[1:] == feat_dim
