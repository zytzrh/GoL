import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PeriodicTable as PT
from rdkit.Chem.rdchem import GetPeriodicTable

from torchdrug.core import Registry as R
from torchdrug.data import feature

import openbabel
from openbabel import pybel
import numpy as np

# atom_types = [6,7,8,9,15,16,17,35,53]
# atom_types_ = [6,7,8,16]


atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vdw_radii = {a: PT.GetRvdw(GetPeriodicTable(), a) for i, a in enumerate(atom_vocab)}
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))

# ********Protein Only*********#
# The hybridization of this atom: 1 for sp, 2 for sp2, 3 for sp3, 4 for sq. planar, 5 for trig. bipy, 6 for octahedral
hyb_vocab = range(7)
heavydegree_vocab = range(8)
heterodegree_vocab = range(8)
# partialcharge_vocab =
AA_vocab = ['ALA', 'ASX', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET',
            'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'SEC', 'VAL', 'TRP', 'XAA', 'TYR', 'GLX']
AA_vocab = {a: i for i, a in enumerate(AA_vocab)}
edge_type_vocab = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4, "Others": 0, "Spatial": 5, "Sequential": 6}
seq_dist_vocab = range(10)

SMARTS = [
    '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
    '[a]',
    '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
    '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
    '[r]'
]
PATTERNS = []
SMARTS_label = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                'ring']
for smarts in SMARTS:
    PATTERNS.append(pybel.Smarts(smarts))



def get_atom_symbol(atomic_number):
    return PT.GetElementSymbol(GetPeriodicTable(), atomic_number)


def onehot(x, vocab, allow_unknown=False, index_=None):
    if index_ is not None:
        index_ = int(index_)
    if x is None and index_ is not None:
        if isinstance(vocab, dict):
            x = list(vocab.keys())[list(vocab.values()).index(index_)]
        else:
            x = vocab[index_]
    if index_ is not None:
        index = index_
    else:
        if x in vocab:
            if isinstance(vocab, dict):
                index = vocab[x]
            else:
                index = vocab.index(x)
        else:
            index = -1

    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature



@R.register("features.atom.protein_default")
def atom_protein_default(atom):
    """Protein-ligand interaction atom feature.
    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return onehot(atom.residue.name, AA_vocab, allow_unknown=True) + \
           onehot(get_atom_symbol(atom.atomicnum), atom_vocab, allow_unknown=True) + \
           onehot(atom.__getattribute__("hyb"), hyb_vocab) + \
           onehot(atom.__getattribute__("heavydegree"), heavydegree_vocab) + \
           onehot(atom.__getattribute__("heterodegree"), heterodegree_vocab) + \
           [atom.__getattribute__("partialcharge")]


@R.register("features.atom.protein_light")
def atom_protein_light(atom):
    """The light-weight protein features to dump.
    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom
        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom
        GetIsAromatic(): whether the atom is aromatic
        IsInRing(): whether the atom is in a ring
    """
    res_id = AA_vocab.get(atom.residue.name, -1)
    atom_symbol = get_atom_symbol(atom.atomicnum)
    atom_id = atom_vocab[atom_symbol] if atom_symbol in atom_vocab else -1
    return [res_id,
            atom_id,
            atom.__getattribute__("hyb"),
            atom.__getattribute__("heavydegree"),
            atom.__getattribute__("heterodegree"),
            atom.__getattribute__("partialcharge")]

@R.register("features.molecule.smarts")
def molecule_smarts(mol):
    """molecular smarts feature.
    Find atoms that match SMARTS patterns.
    Parameters
    ----------
    molecule: pybel.Molecule
    Returns
    -------
    features: np.ndarray
        NxM binary array, where N is the number of atoms in the `molecule`
        and M is the number of patterns. `features[i, j]` == 1.0 if i'th
        atom has j'th property
    """

    if not isinstance(mol, pybel.Molecule):
        raise TypeError('molecule must be pybel.Molecule object, %s was given'
                        % type(mol))

    features = np.zeros((len(mol.atoms), len(PATTERNS)), dtype=np.float32)

    for (pattern_id, pattern) in enumerate(PATTERNS):
        atoms_with_prop = np.array(list(*zip(*pattern.findall(mol))),
                                    dtype=int) - 1
        features[atoms_with_prop, pattern_id] = 1.0
    return features


@R.register("features.bond.type")
def bond_symbol(bond):
    """bond type feature.

    Features:
        GetBondType(): one-hot embedding for the bond type
    """
    return onehot(bond.GetBondType(), bond_type_vocab)
