import torch
from torch import nn

from torchdrug import core, layers
from torchdrug.core import Registry as R


@R.register("models.EGNN")
class EquivariantGraphNeuralNetwork(nn.Module, core.Configurable):
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

    def __init__(self, input_dim, hidden_dim, num_layer=3, node_attr_dim=0, edge_attr_dim=0,
                 activation="ReLU", readout='sum', residual=True, attention=False, normalize=False, tanh=False):
        super(EquivariantGraphNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_layer = num_layer

        self.EGCL_layers = nn.ModuleList()
        for i in range(num_layer):
            node_feat_dim = input_dim if i == 0 else hidden_dim
            self.EGCL_layers.append(layers.EGCL(node_feat_dim, hidden_dim, node_attr_dim, edge_attr_dim,
                                                activation, residual = residual, attention = attention,
                                                normalize = normalize, tanh = tanh))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, node_input, all_loss=None, metric=None, node_attr=None, edge_attr=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            node_input (Tensor): input node representations
            node_attr (Tensor, optional): if specified, use additional node attribute
            edge_attr (Tensor, optional): if specified, use additional edge attribute
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        node_feat = node_input
        coord = graph.atom_coords

        for i in range(self.num_layer):
            node_feat, coord = self.EGCL_layers[i](graph, node_feat, coord, node_attr, edge_attr)
        graph_feat = self.readout(graph, node_feat)

        return {
            "graph_feature": graph_feat,
            "node_feature": node_feat
        }