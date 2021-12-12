import math
import warnings

from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch_scatter import scatter_add, scatter_min, scatter_max

from torchdrug import utils, data
from torchdrug.data import constant, Graph, PackedGraph
from torchdrug.core import Registry as R
from torchdrug.data.rdkit import draw

import pickle
import numpy as np
import openbabel
from openbabel import pybel
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem.rdchem import GetPeriodicTable


plt.switch_backend("agg")

#>>>> part of code in feature.py
from data.featrue import get_atom_symbol, atom_vdw_radii
import data.featrue

class Protein(Graph):
    """
    Protein (Ligand) graph with chemical features.
    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        formal_charge (array_like, optional): formal charges of shape :math:`(|V|,)`
        explicit_hs (array_like, optional): number of explicit hydrogens of shape :math:`(|V|,)`
        chiral_tag (array_like, optional): chirality tags of shape :math:`(|V|,)`
        radical_electrons (array_like, optional): number of radical electrons of shape :math:`(|V|,)`
        atom_map (array_likeb optional): atom mappings of shape :math:`(|V|,)`
        bond_stereo (array_like, optional): bond stereochem of shape :math:`(|E|,)`
        stereo_atoms (array_like, optional): ids of stereo atoms of shape :math:`(|E|,)`
    """

    bond2id = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4, "Others": 0}
    order2id = {1: 1, 2: 2, 3: 3}
    atom2valence = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
    bond2valence = [1, 2, 3, 1.5, 1]
    id2bond = {v: k for k, v in bond2id.items()}
    flag2feat = {0: [1, 0], 1: [0, 1]}
    # empty_mol = Chem.MolFromSmiles("")
    dummy_mol = pybel.readstring("smi", "C")
    dummy_atom = dummy_mol.atoms[0]

    # dummy_bond = dummy_mol.GetBondWithIdx(0)

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, atom_coords=None, atom_flag=None,
                 formal_charge=None,
                 explicit_hs=None, chiral_tag=None, radical_electrons=None, atom_map=None, bond_stereo=None,
                 stereo_atoms=None, atom_radii=None, residue_id=None, atom_id_in_residue=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.bond2id)
        super(Protein, self).__init__(edge_list=edge_list, **kwargs)
        atom_type, bond_type = self._standarize_atom_bond(atom_type, bond_type)
        atom_coords = self._standarize_float_attribute(atom_coords, (self.num_node, 3))
        atom_flag = self._standarize_attribute(atom_flag, self.num_node)
        atom_radii = self._standarize_attribute(atom_radii, self.num_node)
        atom_id_in_residue = self._standarize_attribute(atom_id_in_residue, self.num_node)
        residue_id = self._standarize_attribute(residue_id, self.num_node)

        formal_charge = self._standarize_attribute(formal_charge, self.num_node)
        explicit_hs = self._standarize_attribute(explicit_hs, self.num_node)
        chiral_tag = self._standarize_attribute(chiral_tag, self.num_node)
        radical_electrons = self._standarize_attribute(radical_electrons, self.num_node)
        atom_map = self._standarize_attribute(atom_map, self.num_node)
        bond_stereo = self._standarize_attribute(bond_stereo, self.num_edge)
        stereo_atoms = self._standarize_attribute(stereo_atoms, (self.num_edge, 2))

        with self.node():
            self.atom_type = atom_type
            self.atom_coords = atom_coords
            # self.node_position = atom_coords    # TODO
            self.atom_flag = atom_flag
            self.atom_radii = atom_radii
            self.formal_charge = formal_charge
            self.explicit_hs = explicit_hs
            self.chiral_tag = chiral_tag
            self.radical_electrons = radical_electrons
            self.atom_map = atom_map
            self.atom_id_in_residue = atom_id_in_residue
            self.residue_id = residue_id

        with self.edge():
            self.bond_type = bond_type
            self.bond_stereo = bond_stereo
            self.stereo_atoms = stereo_atoms

    def _standarize_atom_bond(self, atom_type, bond_type):
        if atom_type is None:
            raise ValueError("`atom_type` should be provided")
        if bond_type is None:
            raise ValueError("`bond_type` should be provided")

        atom_type = torch.as_tensor(atom_type, dtype=torch.long, device=self.device)
        bond_type = torch.as_tensor(bond_type, dtype=torch.long, device=self.device)
        return atom_type, bond_type

    def _standarize_attribute(self, attribute, size):
        if attribute is not None:
            attribute = torch.as_tensor(attribute, dtype=torch.long, device=self.device)
        else:
            if isinstance(size, torch.Tensor):
                size = size.tolist()
            attribute = torch.zeros(size, dtype=torch.long, device=self.device)
        return attribute

    def _standarize_float_attribute(self, attribute, size):
        if attribute is not None:
            attribute = torch.as_tensor(attribute, dtype=torch.float32, device=self.device)
        else:
            if isinstance(size, torch.Tensor):
                size = size.tolist()
            attribute = torch.zeros(size, dtype=torch.float32, device=self.device)
        return attribute

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def _check_no_stereo(self):
        if (self.bond_stereo > 0).any():
            warnings.warn("Try to apply masks on molecules with stereo bonds. This may produce invalid molecules. "
                          "To discard stereo information, call `mol.bond_stereo[:] = 0` before applying masks.")

    def _maybe_num_node(self, edge_list):
        if len(edge_list):
            return edge_list[:, :2].max().item() + 1
        else:
            return 0

    @property
    def node_position(self):
        return self.atom_coords

    @classmethod
    def get_bond_type(cls, bond):
        type = bond.GetBondOrder()
        type = cls.order2id.get(type, 0)
        if bond.IsAromatic():
            type = cls.bond2id["AROMATIC"]
        return type

    @classmethod
    def from_molecule(cls, mol,
                      node_feature="protein_default", with_smarts_feature=True,
                      edge_feature=None, graph_feature=None,
                      with_hydrogen=False, with_water=False, with_hetero=True,
                      kekulize=False, save_protein_ligand_flag=0):
        """
        Create a molecule from a RDKit object.
        Parameters:
            mol (pybel.Molecule): molecule
            node_feature (str or list of str, optional): node features to extract
            edge_feature (str or list of str, optional): edge features to extract
            graph_feature (str or list of str, optional): graph features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
            save_protein_ligand_flag (int in [0, 1]): flag indexing the type of the molecule (protein or ligand).
        """

        node_feature = cls._standarize_option(node_feature)
        edge_feature = cls._standarize_option(edge_feature)
        graph_feature = cls._standarize_option(graph_feature)

        atom_type = []
        atom_coords = []
        atom_flag = []
        residue_id = []
        atom_id_in_residue = []
        atoms_selected_index = []
        atoms_selected_id = []
        formal_charge = []
        atom_radii = []

        edge_list = []
        edge_list_tmp = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []

        _node_feature = []
        _edge_feature = []
        _graph_feature = []
        # _protein_ligand_flag = []
        _node_smarts_feature = []

        if mol is None:
            raise ValueError("mol is None")
        # CHANGED
        # if with_hydrogen:
        #     mol.addh()
        # else:
        #     mol.removeh()
        # if kekulize:
        #     pass

        # Process Atom
        idx_map = [-1] * (len(mol.atoms) + 1)
        for i, atom in enumerate(mol):
            assert atom.idx == i + 1
            if with_hydrogen is False and atom.atomicnum <= 1: continue
            if with_water is False and atom.residue.name == "HOH": continue
            idx_map[atom.idx] = len(atoms_selected_index)
            atoms_selected_index.append(i)
            atoms_selected_id.append(atom.idx)

            atom_type.append(atom.atomicnum)
            atom_coords.append(atom.coords)
            atom_flag.append(save_protein_ligand_flag)
            formal_charge.append(atom.formalcharge)
            residue_id.append(atom.residue.idx)
            atom_symbol = get_atom_symbol(atom.atomicnum)
            atom_radii.append(atom_vdw_radii.get(atom_symbol, 1.5))

            if i == 0 or residue_id[-1] != residue_id[-2]:
                atom_id_in_residue.append(atom.atomicnum == 6)
            else:
                atom_id_in_residue.append(atom_id_in_residue[-1] + 1)
            feature = []
            for name in node_feature:
                func = R.get("features.atom.%s" % name)
                feature += func(atom)
            feature += cls.flag2feat[save_protein_ligand_flag]
            _node_feature.append(feature)

        # Process Edge
        for atom in mol:
            h = atom.idx
            if h not in atoms_selected_id: continue
            for natom in openbabel.OBAtomAtomIter(atom.OBAtom):
                t = natom.GetIdx()
                if t not in atoms_selected_id: continue
                bond = openbabel.OBAtom.GetBond(natom, atom.OBAtom)
                type = cls.get_bond_type(bond)
                bond_type.append(type)
                edge_list.append([idx_map[h], idx_map[t], type])
                feature = []
                for name in edge_feature:
                    func = R.get("features.bond.%s" % name)
                    feature += func(bond)
                _edge_feature += [feature]

        for name in graph_feature:
            func = R.get("features.molecule.%s" % name)
            _graph_feature += func(mol)

        # Construct Atom feature in torch format
        atom_type = torch.tensor(atom_type)
        atom_coords = torch.tensor(atom_coords)
        atom_flag = torch.tensor(atom_flag)
        formal_charge = torch.tensor(formal_charge)
        residue_id = torch.tensor(residue_id)
        atom_id_in_residue = torch.tensor(atom_id_in_residue)
        atom_radii = torch.tensor(atom_radii)
        if len(node_feature) > 0:
            _node_feature = torch.tensor(_node_feature)
            # _protein_ligand_flag = torch.tensor(_protein_ligand_flag)
            if with_smarts_feature:
                _node_smarts_feature = R.get("features.molecule.smarts")(mol)[atoms_selected_index]
                _node_smarts_feature = torch.tensor(_node_smarts_feature)
                assert _node_feature.size(0) == _node_smarts_feature.size(0)
                _node_feature = torch.cat((_node_feature, _node_smarts_feature), dim=1)
            if torch.isnan(_node_feature).any():
                raise RuntimeError('Got NaN when calculating features')
        else:
            _node_feature = None

        # Construct Edge&Graph feature in torch format
        edge_list = torch.tensor(edge_list)
        bond_type = torch.tensor(bond_type)

        if len(edge_feature) > 0:
            _edge_feature = torch.tensor(_edge_feature)
        else:
            _edge_feature = None
        if len(graph_feature) > 0:
            _graph_feature = torch.tensor(_graph_feature)
        else:
            _graph_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)

        return cls(edge_list, atom_type, bond_type,
                   atom_coords=atom_coords, atom_flag=atom_flag,
                   formal_charge=formal_charge, explicit_hs=None,
                   chiral_tag=None, radical_electrons=None, atom_map=None,
                   bond_stereo=None, stereo_atoms=None, atom_radii=atom_radii,
                   residue_id=residue_id, atom_id_in_residue=atom_id_in_residue,
                   node_feature=_node_feature, edge_feature=_edge_feature, graph_feature=_graph_feature,
                   num_node=_node_feature.size(0), num_relation=num_relation)

    @classmethod
    def merge_protein_ligand(cls, protein, ligand):
        offset = protein.num_node
        ligand_edge_list = ligand.edge_list.clone()
        ligand_edge_list[:, :2] += offset
        edge_list = torch.cat((protein.edge_list, ligand_edge_list), dim=0)
        atom_type = torch.cat((protein.atom_type, ligand.atom_type), dim=0)
        bond_type = torch.cat((protein.bond_type, ligand.bond_type), dim=0)
        atom_coords = torch.cat((protein.atom_coords, ligand.atom_coords), dim=0)
        atom_flag = torch.cat((protein.atom_flag, ligand.atom_flag), dim=0)
        formal_charge = torch.cat((protein.formal_charge, ligand.formal_charge), dim=0)
        if hasattr(protein, "node_feature"):
            _node_feature = torch.cat((protein.node_feature, ligand.node_feature), dim=0)
        else:
            _node_feature = None
        if hasattr(protein, "edge_feature"):
            _edge_feature = torch.cat((protein.edge_feature, ligand.edge_feature), dim=0)
        else:
            _edge_feature = None

        return cls(edge_list, atom_type, bond_type,
                   atom_coords=atom_coords, atom_flag=atom_flag,
                   formal_charge=formal_charge, explicit_hs=None,
                   chiral_tag=None, radical_electrons=None, atom_map=None,
                   bond_stereo=None, stereo_atoms=None,
                   node_feature=_node_feature, edge_feature=_edge_feature, graph_feature=None,
                   num_node=protein.num_node + ligand.num_node, num_relation=protein.num_relation)

    def node_mask(self, index, compact=False):
        self._check_no_stereo()
        return super(Protein, self).node_mask(index, compact)

    def edge_mask(self, index):
        self._check_no_stereo()
        return super(Protein, self).edge_mask(index)

    @property
    def num_atom(self):
        """Number of atoms."""
        return self.num_node

    @property
    def num_bond(self):
        """Number of bonds."""
        return self.num_edge

    @utils.cached_property
    def explicit_valence(self):
        bond2valence = torch.tensor(self.bond2valence, device=self.device)
        explicit_valence = scatter_add(bond2valence[self.edge_list[:, 2]], self.edge_list[:, 0], dim_size=self.num_node)
        return explicit_valence.round().long()

    @utils.cached_property
    def is_valid(self):
        """A coarse implementation of valence check."""
        # TODO: cross-check by any domain expert
        atom2valence = torch.tensor(float["nan"]).repeat(constant.NUM_ATOM)
        for k, v in self.atom2valence:
            atom2valence[k] = v
        atom2valence = torch.as_tensor(atom2valence, device=self.device)

        max_atom_valence = atom2valence[self.atom_type]
        # special case for nitrogen
        pos_nitrogen = (self.atom_type == 7) & (self.formal_charge == 1)
        max_atom_valence[pos_nitrogen] = 4
        if torch.isnan(max_atom_valence).any():
            index = torch.isnan(max_atom_valence).nonzero()[0]
            raise ValueError("Fail to check valence. Unknown atom type %d" % self.atom_type[index])

        is_valid = (self.explicit_valence <= max_atom_valence).all()
        return is_valid


class PackedProtein(PackedGraph, Protein):
    """
    Container for molecules with variadic sizes.
    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        num_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edge_list`
        num_edges (array_like, optional): number of edges in each graph
        num_relation (int, optional): number of relations
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
            If not provided, nodes in `edge_list` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edge_list` should be absolute index, i.e., the index in the packed graph.
    """

    unpacked_type = Protein

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, atom_coords=None, atom_flag=None,
                 num_nodes=None, num_edges=None, offsets=None, **kwargs):
        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.bond2id)
        super(PackedProtein, self).__init__(edge_list=edge_list, num_nodes=num_nodes, num_edges=num_edges,
                                            atom_coords=atom_coords,
                                            atom_flag=atom_flag, offsets=offsets,
                                            atom_type=atom_type, bond_type=bond_type, **kwargs)

    @property
    def node_position(self):
        return self.atom_coords

    @utils.cached_property
    def is_valid(self):
        """A coarse implementation of valence check."""
        # TODO: cross-check by any domain expert
        atom2valence = torch.tensor(float("nan")).repeat(118)
        for k, v in self.atom2valence.items():
            atom2valence[k] = v
        atom2valence = torch.as_tensor(atom2valence, device=self.device)

        max_atom_valence = atom2valence[self.atom_type]
        # special case for nitrogen
        pos_nitrogen = (self.atom_type == 7) & (self.formal_charge == 1)
        max_atom_valence[pos_nitrogen] = 4
        if torch.isnan(max_atom_valence).any():
            index = torch.isnan(max_atom_valence).nonzero()[0]
            raise ValueError("Fail to check valence. Unknown atom type %d" % self.atom_type[index])

        is_valid = self.explicit_valence <= max_atom_valence
        is_valid = scatter_min(is_valid.long(), self.node2graph, dim_size=self.batch_size)[0].bool()
        return is_valid


    @classmethod
    def from_molecule(cls, mols, node_feature="protein_default", edge_feature=None, graph_feature=None,
                      with_hydrogen=False, kekulize=False, save_protein_ligand_flags=None):
        """
        Create a packed molecule from a list of RDKit objects.
        Parameters:
            mols (list of pybel.Molecule): molecules
            node_feature (str or list of str, optional): node features to extract
            edge_feature (str or list of str, optional): edge features to extract
            graph_feature (str or list of str, optional): graph features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
            save_protein_ligand_flag (int in [0, 1]): list of flag indexing the type of the molecule (protein or ligand).
        """
        node_feature = cls._standarize_option(node_feature)
        edge_feature = cls._standarize_option(edge_feature)
        graph_feature = cls._standarize_option(graph_feature)

        atom_type = []
        atom_coords = []
        atom_flag = []
        formal_charge = []

        edge_list = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []

        _node_feature = []
        _edge_feature = []
        _graph_feature = []
        # _protein_ligand_flag = []
        _node_smarts_feature = []
        num_nodes = []
        num_edges = []

        for mol_idx, mol in enumerate(mols):
            if mol is None:
                continue

            if with_hydrogen:
                mol.addh()
            else:
                mol.removeh()
            if kekulize:
                pass

            heavy_atoms = []
            for i, atom in enumerate(mol):
                if atom.atomicnum > 1:  #TODO: only consider non-H
                    atom_type.append(atom.atomicnum)
                    atom_coords.append(atom.coords)
                    atom_flag.append(save_protein_ligand_flags[mol_idx])
                    heavy_atoms.append(i)
                    formal_charge.append(atom.formalcharge)
                    feature = []
                    for name in node_feature:
                        func = R.get("features.atom.%s" % name)
                        feature += func(atom)
                    feature += cls.flag2feat[save_protein_ligand_flags[mol_idx]]
                    _node_feature.append(feature)

            # _protein_ligand_flag.append(save_protein_ligand_flags[mol_idx] * np.ones((len(heavy_atoms), 1)))
            _node_smarts_feature.append(R.get("features.molecule.smarts")(mol)[heavy_atoms])    #TODO

            idx_map = [-1] * (len(mol.atoms) + 1)
            idx_new = 0
            edge_list_tmp = []
            for atom in mol:
                h = atom.idx
                if atom.atomicnum == 1:
                    continue
                idx_map[h] = idx_new
                idx_new += 1
                for natom in openbabel.OBAtomAtomIter(atom.OBAtom):
                    if natom.GetAtomicNum() == 1:
                        continue
                    t = natom.GetIdx()
                    bond = openbabel.OBAtom.GetBond(natom, atom.OBAtom)
                    type = bond.GetBondOrder()
                    type = cls.order2id.get(type, 0)
                    if bond.IsAromatic():
                        type = cls.bond2id["AROMATIC"]
                    edge_list_tmp += [[h, t, type]]
                    bond_type.append(type)
                    feature = []
                    for name in edge_feature:
                        func = R.get("features.bond.%s" % name)
                        feature += func(bond)
                    _edge_feature += [feature]

            for h, t, type in edge_list_tmp:
                h_, t_ = idx_map[h], idx_map[t]
                assert (h_ != -1 and t_ != -1)
                edge_list.append([h_, t_, type])

            for name in graph_feature:
                func = R.get("features.molecule.%s" % name)
                _graph_feature += func(mol)

            num_nodes.append(len(heavy_atoms))
            num_edges.append(len(edge_list_tmp))

        atom_type = torch.tensor(atom_type)
        atom_coords = torch.tensor(atom_coords)
        atom_flag = torch.tensor(atom_flag)
        formal_charge = torch.tensor(formal_charge)
        if len(node_feature) > 0:
            _node_feature = torch.tensor(_node_feature)
            # _protein_ligand_flag = torch.tensor(np.concatenate(_protein_ligand_flag, axis=0))
            _node_smarts_feature = torch.tensor(np.concatenate(_node_smarts_feature, axis=0))
            assert _node_feature.size(0) == _node_smarts_feature.size(0)
            _node_feature = torch.cat((_node_feature, _node_smarts_feature), dim=1)
            if torch.isnan(_node_feature).any():
                raise RuntimeError('Got NaN when calculating features')
        else:
            _node_feature = None

        edge_list = torch.tensor(edge_list)
        bond_type = torch.tensor(bond_type)
        bond_stereo = None
        stereo_atoms = None

        if len(edge_feature) > 0:
            _edge_feature = torch.tensor(_edge_feature)
        else:
            _edge_feature = None
        if len(graph_feature) > 0:
            _graph_feature = torch.tensor(_graph_feature)
        else:
            _graph_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)

        return cls(edge_list, atom_type, bond_type,
                   atom_coords=atom_coords, atom_flag=atom_flag,
                   formal_charge=formal_charge, explicit_hs=None,
                   chiral_tag=None, radical_electrons=None, atom_map=None,
                   bond_stereo=None, stereo_atoms=None,
                   node_feature=_node_feature, edge_feature=_edge_feature, graph_feature=_graph_feature,
                   num_nodes=num_nodes, num_edges=num_edges, num_relation=num_relation)

    def node_mask(self, index, compact=False):
        self._check_no_stereo()
        return super(PackedProtein, self).node_mask(index, compact)

    def edge_mask(self, index):
        self._check_no_stereo()
        return super(PackedProtein, self).edge_mask(index)




Protein.packed_type = PackedProtein
