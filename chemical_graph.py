from __future__ import annotations

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, BondType
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data


class MolecularGraph:
    VALID_ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    VALID_BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    UNKNOWN_ATOM = "X"

    def __init__(self, label: str, smiles: str):
        self._label = label
        self._smiles = smiles
        self._geometry = None

    def __get_valid_elements(self, is_stripped: bool = True) -> list[str]:
        return self.VALID_ELEMENTS if is_stripped else self.VALID_ELEMENTS + ["H"]

    def __normalize_mass(self, atom: Atom) -> float:
        min_mass = min([Atom(symbol).GetMass() for symbol in self.VALID_ELEMENTS])
        max_mass = max([Atom(symbol).GetMass() for symbol in self.VALID_ELEMENTS])

        return (atom.GetMass() - min_mass) / (max_mass - min_mass)

    def __normalize_van_der_waals_radius(self, atom: Atom) -> float:
        min_radius = min([Chem.GetPeriodicTable().GetRvdw(Atom(symbol).GetAtomicNum()) for symbol in self.VALID_ELEMENTS])
        max_radius = max([Chem.GetPeriodicTable().GetRvdw(Atom(symbol).GetAtomicNum()) for symbol in self.VALID_ELEMENTS])

        return (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - min_radius) / (max_radius - min_radius)

    def _encode(self, atom_symbol: str | int, valid_symbols: list[str | int], fallback_value: str | int = UNKNOWN_ATOM) -> list[int]:
        if atom_symbol not in valid_symbols:
            atom_symbol = fallback_value

        return [int(atom_symbol == element) for element in valid_symbols]

    def _extract_features(self, atom: Atom, is_chiral: bool = True, is_stripped: bool = True) -> np.array:
        atom_encoded = self._encode(atom.GetSymbol(), self.__get_valid_elements(is_stripped))
        heavy_neighbors_encoded = self._encode(atom.GetDegree(), [0, 1, 2, 3, 4], fallback_value="MoreThanFour")
        formal_charge_encoded = self._encode(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3], fallback_value="OverThree")
        hybridisation_encoded = self._encode(atom.GetHybridization(), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"], fallback_value="Unknown")
        aromaticity_encoded = [int(atom.GetIsAromatic())]
        ring_info_encoded = [int(atom.IsInRing())]
        mass_normalized_norm = [self.__normalize_mass(atom)]
        van_der_waals_radius_norm = [self.__normalize_van_der_waals_radius(atom)]

        feature_vec = atom_encoded + heavy_neighbors_encoded + formal_charge_encoded + hybridisation_encoded + aromaticity_encoded + ring_info_encoded + mass_normalized_norm + van_der_waals_radius_norm

        if is_chiral:
            chirality_encoded = self._encode(atom.GetChiralTag(), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"], fallback_value="CHI_OTHER")
            feature_vec += chirality_encoded

        if is_stripped:
            hydrogens_encoded = self._encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], fallback_value="MoreThanFour")
            feature_vec += hydrogens_encoded

        return np.array(feature_vec)

    def _extract_bond_features(self, bond: Bond) -> np.array:
        bond_type_encoded = self._encode(bond.GetBondType(), self.VALID_BONDS, fallback_value=BondType.OTHER)
        bond_conjugation_encoded = [int(bond.GetIsConjugated())]
        bond_ring_info_encoded = [int(bond.IsInRing())]
        bond_stereo_encoded = self._encode(bond.GetStereo(), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOZ"], fallback_value="STEREOOTHER")

        feature_vec = bond_type_encoded + bond_conjugation_encoded + bond_ring_info_encoded + bond_stereo_encoded

        return np.array(feature_vec)

    def compute_geometry(self):
        mol = Chem.MolFromSmiles(self._smiles)
        nodes, edges = mol.GetNumAtoms(), 2 * mol.GetNumBonds()

        # https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
        support_mol = Chem.MolFromSmiles("O=O")
        node_features_size = len(self._extract_features(support_mol.GetAtomWithIdx(0)))
        bond_features_size = len(self._extract_bond_features(support_mol.GetBondBetweenAtoms(0, 1)))

        node_feature_matrix = np.zeros((nodes, node_features_size))
        edge_feature_matrix = np.zeros((edges, bond_features_size))

        for atom in mol.GetAtoms():
            node_feature_matrix[atom.GetIdx(), :] = self._extract_features(atom)

        r, c = np.nonzero(GetAdjacencyMatrix(mol))
        tr, tc = torch.from_numpy(r.astype(np.int64)).to(torch.long), torch.from_numpy(c.astype(np.int64)).to(torch.long)

        edge_matrix = torch.stack([tr, tc], dim=0)

        for (id, (i, j)) in enumerate(zip(r, c)):
            bond = mol.GetBondBetweenAtoms(i, j)
            edge_feature_matrix[id:] = self._extract_bond_features(bond)

        node_feature_matrix = torch.tensor(node_feature_matrix, dtype=torch.float)
        edge_feature_matrix = torch.tensor(edge_feature_matrix, dtype=torch.float)

        label_tensor = torch.tensor([self._label], dtype=torch.float)

        self._geometry = Data(x=node_feature_matrix, edge_index=edge_matrix, edge_attr=edge_feature_matrix, y=label_tensor)

    def get_geometry(self) -> Data:
        if self._geometry is None:
            self.compute_geometry()
        return self._geometry
