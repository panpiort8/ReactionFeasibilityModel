from typing import List

import dgl
import torch

ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "W",
    "Ru",
    "Nb",
    "Re",
    "Te",
    "Rh",
    "Ta",
    "Tc",
    "Ba",
    "Bi",
    "Hf",
    "Mo",
    "U",
    "Sm",
    "Os",
    "Ir",
    "Ce",
    "Gd",
    "Ga",
    "Cs",
]


def disjoint_union_plain(graphs: List[dgl.DGLGraph]) -> dgl.DGLGraph:
    """
    Creates a disjoint union of multiple homogeneous DGLGraphs.

    This method takes a list of homogeneous DGLGraphs that have matching feature
    keys and feature shapes, and produces a single graph that is their disjoint
    union. The node and edge features of the resulting graph are concatenated
    from the input graphs. Each input graph is offset in terms of node IDs to
    ensure disjoint node spaces across the graphs.

    Args:
        graphs: list of dgl.DGLGraph
            A list of homogeneous DGLGraphs. All graphs must have the same node
            and edge feature keys, and the associated feature tensors must have
            matching shapes.

    Returns:
        dgl.DGLGraph:
            A single DGLGraph that represents the disjoint union of the input graphs.
    """

    device = graphs[0].device
    offsets = []
    cum = 0
    for g in graphs:
        offsets.append(cum)
        cum += g.num_nodes()

    all_u, all_v = [], []
    node_feats = {k: [] for k in graphs[0].ndata.keys()}
    edge_feats = {k: [] for k in graphs[0].edata.keys()}

    for off, g in zip(offsets, graphs):
        u, v = g.edges()
        all_u.append(u + off)
        all_v.append(v + off)
        for k in node_feats:
            node_feats[k].append(g.ndata[k])
        for k in edge_feats:
            edge_feats[k].append(g.edata[k])

    u = torch.cat(all_u).to(device)
    v = torch.cat(all_v).to(device)
    big = dgl.graph((u, v), num_nodes=cum, device=device)

    for k, parts in node_feats.items():
        big.ndata[k] = torch.cat(parts, dim=0)
    for k, parts in edge_feats.items():
        big.edata[k] = torch.cat(parts, dim=0)

    return big
