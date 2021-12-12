import os
import glob

import torch_scatter
from openbabel import pybel
from rdkit import Chem
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from os import walk

from torch.utils import data as torch_data
from torch_cluster import knn_graph
from torch_scatter import scatter_add
import torch

from torchdrug import core, data, utils
from data.protein import Protein




class LigandGenerationDataset(torch_data.Dataset, core.Configurable):
    def __init__(self):
        self.ligand_graphs = []
        self.ligand_ids = []
        self.protein_graphs = []
        self.protein_ids = []
        self.aux_records = []

    def load_protein_graph(self, protein_file, box_center, box_size=(20, 20, 20)):
        pybel_protein = next(pybel.readfile('pdb', protein_file))
        protein_graph = Protein.from_molecule(pybel_protein,
                                                  node_feature="protein_default",
                                                  with_smarts_feature=True,
                                                  edge_feature=None,
                                                  graph_feature=None,
                                                  with_hydrogen=True,   #CHANGED
                                                  with_water=False,
                                                  with_hetero=True, #TODO: not implemented
                                                  kekulize=False,
                                                  save_protein_ligand_flag=1,)  #TODO
        atom_coords_tmp = protein_graph.atom_coords - box_center
        atom_in_box = (torch.abs(atom_coords_tmp) < torch.tensor(box_size)/2).sum(1) == 3

        binding_site_graph = protein_graph.subgraph(atom_in_box)
        self.protein_graphs.append(binding_site_graph)


    def load_ligand_graph(self, ligand_file, with_hydrogen=False):
        rdkit_ligand = Chem.SDMolSupplier(ligand_file, sanitize=False).__next__()  # TODO
        ligand_graph = data.Molecule.from_molecule(rdkit_ligand,
                                                   node_feature="symbol",  # TODO: use complex feature
                                                   edge_feature=None)
        ligand_graph.node_feature = ligand_graph.node_feature.to(torch.float32)

        # get pos
        assert rdkit_ligand.GetNumConformers() == 1
        atom_coords = torch.tensor(rdkit_ligand.GetConformer(0).GetPositions(), dtype=torch.float32)
        assert atom_coords.size(0) == ligand_graph.num_node
        with ligand_graph.node():
            ligand_graph.atom_coords = atom_coords

        if not with_hydrogen:
            ligand_graph = ligand_graph.subgraph(ligand_graph.atom_type != 1)

        self.ligand_graphs.append(ligand_graph)

    def getKNN(self, ligand_graph, protein_graph):
        ligand_atom_feature = ligand_graph.node_feature
        protein_atom_feature = protein_graph.node_feature
        ligand_feature_dim = ligand_atom_feature.shape[1]
        protein_feature_dim = protein_atom_feature.shape[1]
        ligand_atom_feature = torch.cat([torch.zeros(ligand_graph.num_node, protein_feature_dim),
                                           ligand_atom_feature], dim=-1)
        protein_atom_feature = torch.cat([protein_atom_feature,
                                          torch.zeros(protein_graph.num_node, ligand_feature_dim)], dim=-1)
        complex_atom_feature = torch.cat([ligand_atom_feature, protein_atom_feature], dim=0)


        ligand_atom_coords = ligand_graph.atom_coords
        protein_atom_coords = protein_graph.atom_coords
        complex_atom_coords = torch.cat([ligand_atom_coords, protein_atom_coords])
        edge_list = knn_graph(complex_atom_coords, k=48)


        complex_graph = data.Graph(edge_list=edge_list, num_node=complex_atom_feature.shape[0],
                          node_feature=complex_atom_feature)
        with complex_graph.node():
            complex_graph.node_position = complex_atom_coords
        return complex_graph


    def __getitem__(self, index):
        ligand_graph = self.ligand_graphs[index]
        protein_graph = self.protein_graphs[index]

        return {
            "ligand_graph": ligand_graph,
            "protein_graph": protein_graph
        }

    def __getitem_tmp__(self, index):
        ligand_graph = self.ligand_graphs[index]
        mask_node_index = torch.rand(ligand_graph.num_node) > torch.rand(1)
        mask_coord = ligand_graph.atom_coords[mask_node_index]
        mask_type = ligand_graph.atom_type[mask_node_index]

        # Get Complex Graph
        ligand_graph_masked = ligand_graph.subgraph(~mask_node_index)
        protein_graph = self.protein_graphs[index]
        complex_graph = self.getKNN(ligand_graph_masked, protein_graph)

        # Get Frontier (based on atom bond?)
        old_edge_num = torch_scatter.scatter_add(src=torch.ones(ligand_graph.num_edge),
                                                 index=ligand_graph.edge_list[:,0],
                                                 dim_size=ligand_graph.num_node)
        old_edge_num = old_edge_num[~mask_node_index]
        new_edge_num = torch_scatter.scatter_add(src=torch.ones(ligand_graph_masked.num_edge),
                                                 index=ligand_graph_masked.edge_list[:,0],
                                                 dim_size=ligand_graph_masked.num_node)
        is_frontier = old_edge_num > new_edge_num

        return {
            "complex_graph": complex_graph,
            "target_coord": mask_coord,
            "target_type": mask_type,
            "is_frontier": is_frontier,
        }

    def __len__(self):
        return len(self.protein_graphs)

    @property
    def node_feature_dim_protein(self):
        """Dimension of proetin node features."""
        return self.protein_graphs[0].node_feature.shape[-1]

    @property
    def node_feature_dim_ligand(self):
        """Dimension of ligand node features."""
        return self.ligand_graphs[0].node_feature.shape[-1]


class CrossDockedSubset(LigandGenerationDataset):
    def __init__(self, subset_folder):
        super(CrossDockedSubset, self).__init__()
        self.subset_folder = subset_folder

        self.protein_files = []
        self.ligand_files = []


        self.load_path(self.subset_folder)
        self.load_data()


    def load_path(self, subset_folder):
        for (dir_path, dir_names, file_names) in walk(subset_folder):
            # print(dir_path)
            # print(dir_names)
            # print(file_name)
            # break
            for file_name in file_names:
                if file_name.split(".")[-1] == "sdf":
                    split_names = file_name.split(".")[0].split("_")
                    protein_id = "_".join(split_names[:3])
                    ligand_id = "_".join(split_names[3:6])
                    aux_record = "_".join(split_names[6:])

                    self.protein_ids.append(protein_id)
                    self.ligand_ids.append(ligand_id)
                    self.aux_records.append(aux_record)

                    protein_file = os.path.join(dir_path, f"{protein_id}.pdb")
                    ligand_file = os.path.join(dir_path, file_name)
                    assert os.path.exists(protein_file)
                    self.protein_files.append(protein_file)
                    self.ligand_files.append(ligand_file)


    def load_data(self):
        for i in tqdm(range(len(self.protein_files))):
            ligand_file = self.ligand_files[i]
            self.load_ligand_graph(ligand_file)
            ligand_center = self.ligand_graphs[-1].atom_coords.mean(dim=0)

            protein_file = self.protein_files[i]
            self.load_protein_graph(protein_file, box_center=ligand_center)




def test():
    # dataset = CrossDockedSubset("/home/mila/y/yangtian.zhang/scratch/dataset/crossdocked_subset/crossdocked_subset")
    dataset = CrossDockedSubset("/home/mila/y/yangtian.zhang/scratch/dataset/crossdocked_subset/crossdocked_subset/1433B_HUMAN_1_240_pep_0")
    a = dataset[0]
    print(a)

if __name__ == "__main__":
    test()