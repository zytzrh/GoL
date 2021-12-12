from layers.docking_layer import EGCL, IntraAttention

import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R


@R.register("models.IntraEGNN")
class IntraEquivariantGraphNeuralNetwork(nn.Module, core.Configurable):
    """
    E(n) Equivariant Graph Neural Network.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        num_layer (int, optional): number of hidden layers
        activation (str or function, optional): activation function
        residual (bool): whether to use skip connection for each layers
        attention (bool): whether to use attention
        normalize (bool): whether to normalize the coordinate messages
        tanh (bool): whether to apply an tanh activation on the output of weight function
    """

    def __init__(self, input_dim_protein, input_dim_ligand,
                 hidden_dim, num_layer=3):
        super(IntraEquivariantGraphNeuralNetwork, self).__init__()

        self.input_dim_protein = input_dim_protein
        self.input_dim_ligand = input_dim_ligand
        self.num_layer = num_layer

        self.inter_layer_protein = nn.ModuleList()
        self.inter_layer_ligand = nn.ModuleList()
        self.intra_layer = nn.ModuleList()
        for i in range(num_layer):
            node_dim_protein  = input_dim_protein if i == 0 else hidden_dim
            node_dim_ligand = input_dim_ligand if i ==0 else hidden_dim

            self.inter_layer_protein.append(EGCL(node_dim_protein, hidden_dim))
            self.inter_layer_ligand.append(layers.GraphIsomorphismConv(node_dim_ligand, hidden_dim))
            self.intra_layer.append(IntraAttention(hidden_dim, hidden_dim))


    def forward(self, graph_protein, input_protein, graph_ligand, input_ligand):
        node_feat_protein = input_protein
        node_feat_ligand = input_ligand
        coord_protein = graph_protein.atom_coords

        for i in range(self.num_layer):
            node_feat_protein, coord_protein = self.inter_layer_protein[i](graph_protein, node_feat_protein, coord_protein)
            node_feat_ligand = self.inter_layer_ligand[i](graph_ligand, node_feat_ligand)
            node_feat_protein, node_feat_ligand = self.intra_layer[i](graph_protein, node_feat_protein,
                                                                   graph_ligand, node_feat_ligand,
                                                                   graph_intra=None)
        return {
            "feature_protein": node_feat_protein,
            "feature_ligand": node_feat_ligand
        }
