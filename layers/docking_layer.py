from utils.utils import getIndex

import functools

import torch
from torch import nn
from torch._C import dtype
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_scatter.composite import scatter_softmax

from torchdrug import data, layers, utils
from torchdrug.layers import functional

class IntraAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(IntraAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_layer = layers.GraphAttentionConv(self.input_dim, self.hidden_dim)

    def get_graph_intra(self, graph_protein, graph_ligand):
        node_num_protein = graph_protein.num_nodes
        node_num_ligand = graph_ligand.num_nodes
        node_num_protein_all = graph_protein.num_node
        node_num_ligand_all = graph_ligand.num_node

        index_protein, index_ligand = getIndex(node_num_protein, node_num_ligand)
        index_ligand += node_num_protein_all    # add offset
        new_node_in = torch.cat([index_protein, index_ligand])
        new_node_out = torch.cat([index_ligand, index_protein])
        new_edge_list = torch.vstack([new_node_in, new_node_out]).transpose(0,1)

        return data.Graph(edge_list=new_edge_list, edge_weight=None,
                          num_node=node_num_protein_all+node_num_ligand_all)


    def forward(self, graph_protein, node_feat_protein, graph_ligand, node_feat_ligand, graph_intra):
        if graph_intra is None:
            graph_intra = self.get_graph_intra(graph_protein, graph_ligand)
        node_feat = torch.cat([node_feat_protein, node_feat_ligand])
        output = self.attention_layer(graph_intra, node_feat)
        return output[:len(node_feat_protein)], output[len(node_feat_protein):]


class EGCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, in_dim, hid_dim, node_attr_dim=0, edge_attr_dim=0, activation="ReLU",
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        in_edge_dim = in_dim * 2
        edge_coor_dim = 1

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim + edge_coor_dim + edge_attr_dim, hid_dim),
            self.activation,
            nn.Linear(hid_dim, hid_dim),
            self.activation
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hid_dim + in_dim + node_attr_dim, hid_dim),
            self.activation,
            nn.Linear(hid_dim, hid_dim)
        )

        weight_layer = nn.Linear(hid_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(weight_layer.weight, gain=0.001)
        if self.tanh:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                self.activation,
                weight_layer,
                nn.Tanh()
            )
        else:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                self.activation,
                weight_layer
            )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hid_dim, 1),
                nn.Sigmoid()
            )

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)

        return result

    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)
        count = data.new_full(result_shape, 0)
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, torch.ones_like(data))

        return result / count.clamp(min=1)

    def edge_function(self, src_node_feat, tgt_node_feat, radial, edge_attr):
        if edge_attr is None:
            edge_feat_in = torch.cat([src_node_feat, tgt_node_feat, radial], dim=1)
        else:
            edge_feat_in = torch.cat([src_node_feat, tgt_node_feat, radial, edge_attr], dim=1)

        edge_feat_out = self.edge_mlp(edge_feat_in)
        if self.attention:
            att_weight = self.att_mlp(edge_feat_out)
            edge_feat_out = edge_feat_out * att_weight

        return edge_feat_out

    def coord_function(self, coord, edge_list, coord_diff, edge_feat):
        node_in = edge_list[:, 0]
        node_out = edge_list[:, 1]
        weighted_trans = coord_diff * self.coord_mlp(edge_feat)

        if self.coords_agg == 'sum':
            agg_trans = self.unsorted_segment_sum(weighted_trans, node_out, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg_trans = self.unsorted_segment_mean(weighted_trans, node_out, num_segments=coord.size(0))
        else:
            raise NotImplementedError('Aggregation method {} is not implemented'.format(self.coords_agg))
        coord += agg_trans

        return coord

    def node_function(self, node_feat, edge_list, edge_feat, node_attr):
        node_in = edge_list[:, 0]
        node_out = edge_list[:, 1]
        agg_edge_feat = self.unsorted_segment_sum(edge_feat, node_out, num_segments=node_feat.size(0))

        if node_attr is not None:
            node_feat_in = torch.cat([node_feat, agg_edge_feat, node_attr], dim=1)
        else:
            node_feat_in = torch.cat([node_feat, agg_edge_feat], dim=1)

        node_feat_out = self.node_mlp(node_feat_in)
        if self.residual:
            if node_feat.size(1) == node_feat_out.size(1):
                node_feat_out = node_feat + node_feat_out

        return node_feat_out

    def coord2radial(self, edge_list, coord, epsilon=1e-6):
        node_in = edge_list[:, 0]
        node_out = edge_list[:, 1]
        coord_diff = coord[node_out] - coord[node_in]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, graph, node_feat, coord, node_attr=None, edge_attr=None):
        node_in = graph.edge_list[:, 0]
        node_out = graph.edge_list[:, 1]
        radial, coord_diff = self.coord2radial(graph.edge_list, coord)

        edge_feat = self.edge_function(node_feat[node_in], node_feat[node_out], radial, edge_attr)
        coord = self.coord_function(coord, graph.edge_list, coord_diff, edge_feat)
        node_feat = self.node_function(node_feat, graph.edge_list, edge_feat, node_attr)

        return node_feat, coord