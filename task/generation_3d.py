import torch
import torch_scatter
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
from torch_cluster import knn_graph, knn

from torchdrug import core, tasks, data, layers
from data.featrue import atom_vocab

class LigandGeneration(tasks.Task, core.Configurable):
    def __init__(self, complex_model, spatial_model):
        super().__init__()
        self.complex_model = complex_model
        self.spatial_model = spatial_model
        # TODO: shape problem
        self.frontier_model = layers.MultiLayerPerceptron(complex_model.output_dim, hidden_dims=[128, 1], activation="leaky_relu")
        self.classifier_head = layers.MultiLayerPerceptron(spatial_model.output_dim, hidden_dims=[128, len(atom_vocab) + 1], activation="leaky_relu")

    # Consider batch? Directly use coord of complex as negative sampling
    def sample_negative_points(self, complex_graph, sample_num):
        complex_coords = complex_graph.atom_coords
        random_index = torch.randint(low=0, high=len(complex_coords), size=(sample_num,))
        batch_negative = complex_graph.node2graph[random_index] #TODO: maybe wrong
        coords_negative = complex_coords[random_index]

        random_offset = torch.randn_like(coords_negative)
        coords_negative += random_offset

        return coords_negative, batch_negative


    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        pred, target = self.predict_and_target(batch)
        pred_type, pred_none, pred_is_frontier = pred
        target_type, target_none, target_is_frontier = target

        loss_BCE = F.binary_cross_entropy(pred_none, target_none)
        loss_CAT = F.cross_entropy(pred_type, target_type)
        loss_Frontier = F.binary_cross_entropy_with_logits(pred_is_frontier, target_is_frontier)

        all_loss += loss_BCE
        all_loss += loss_CAT
        all_loss += loss_Frontier

        metric["BCE Loss"] = loss_BCE
        metric["CAT Loss"] = loss_CAT
        metric["Frontier Classification Loss"] = loss_Frontier



        return all_loss, metric



    # # batch support?
    # def forward2(self, batch):
    #     complex_graph = batch["complex_graph"]
    #     target_coord = batch["target_coord"]
    #     target_type = batch["target_type"]
    #     is_frontier = batch["is_frontier"]
    #
    #     context_output = self.complex_model(complex_graph)
    #     context_encoding = context_output["node_feature"]


    def getKNN(self, ligand_graph, protein_graph):
        # Complex Atom Feature
        ligand_atom_feature = ligand_graph.node_feature
        protein_atom_feature = protein_graph.node_feature
        ligand_feature_dim = ligand_atom_feature.shape[1]
        protein_feature_dim = protein_atom_feature.shape[1]
        ligand_atom_feature = torch.cat([torch.zeros((ligand_graph.num_node, protein_feature_dim,), device=self.device),
                                           ligand_atom_feature], dim=-1)
        protein_atom_feature = torch.cat([protein_atom_feature,
                                          torch.zeros((protein_graph.num_node, ligand_feature_dim), device=self.device)], dim=-1)
        complex_atom_feature = torch.cat([ligand_atom_feature, protein_atom_feature], dim=0)

        # Complex Atom Coords
        ligand_atom_coords = ligand_graph.atom_coords
        protein_atom_coords = protein_graph.atom_coords
        complex_atom_coords = torch.cat([ligand_atom_coords, protein_atom_coords])

        # Complex Batch
        complex_batch = torch.cat([ligand_graph.node2graph, protein_graph.node2graph])


        # Return Packed Graph
        edge_list = knn_graph(complex_atom_coords, k=48, batch=complex_batch, loop=False)
        complex_graph = data.Graph(edge_list=edge_list, num_node=complex_atom_feature.shape[0],
                          node_feature=complex_atom_feature)
        with complex_graph.node():
            complex_graph.node_position = complex_atom_coords
            complex_graph.atom_coords = complex_graph.node_position
            complex_graph.is_ligand = torch.zeros((ligand_graph.num_node+protein_graph.num_node,),
                                                  dtype=torch.bool, device=self.device)
            complex_graph.is_ligand[:ligand_graph.num_node] = True
        return complex_graph.split(node2graph=complex_batch)


    def getSpatialEmbedding(self, context_coords, context_batch, context_embedding,
                            predicted_coords, predicted_batch, predicted_initial_embedding=None):
        predicted_node_num = len(predicted_batch)
        context_node_num = len(context_batch)
        if predicted_initial_embedding is None:
            predicted_initial_embedding = torch.zeros((predicted_node_num, context_embedding.shape[1]), device=self.device)
        # Construct Spatial Graph
        edge_list = knn(x=context_coords, y=predicted_coords, k=32, #TODO
                        batch_x=context_batch, batch_y=predicted_batch)
        edge_list[:, 0] += predicted_node_num           # offset
        spatial_graph =  data.Graph(edge_list=edge_list, num_node=predicted_node_num+context_node_num,
                                    node_feature=torch.cat([predicted_initial_embedding, context_embedding], dim=0))
        with spatial_graph.node():
            spatial_graph.node_position = torch.cat([predicted_coords, context_coords], dim=0)
            spatial_graph.atom_coords = spatial_graph.node_position
        # Get Spatial Embedding
        spatial_embedding = self.spatial_model(spatial_graph, spatial_graph.node_feature)["node_feature"][:predicted_node_num]
        return spatial_embedding


    def predict_and_target(self, batch, all_Loss=None, metric=None):
        ligand_graph = batch["ligand_graph"]
        protein_graph = batch["protein_graph"]


        # Random mask
        mask_node_index = torch.rand(ligand_graph.num_node) > torch.rand(1)
        mask_coord = ligand_graph.atom_coords[mask_node_index]
        mask_type = ligand_graph.atom_type[mask_node_index]
        mask_batch = ligand_graph.node2graph[mask_node_index]


        # Get Complex Graph
        ligand_graph_masked = ligand_graph.subgraph(~mask_node_index)
        complex_graph = self.getKNN(ligand_graph_masked, protein_graph)


        # Context Encoder
        context_embedding = self.complex_model(complex_graph, complex_graph.node_feature)["node_feature"]


        # Spatial Prediction
        negative_coords, negative_batch = self.sample_negative_points(complex_graph, len(mask_coord))
        spatial_embedding = self.getSpatialEmbedding(context_coords=complex_graph.atom_coords,
                                                     context_batch=complex_graph.node2graph,
                                                     context_embedding=context_embedding,
                                                     predicted_coords=torch.cat([mask_coord, negative_coords], dim=0),
                                                     predicted_batch=torch.cat([mask_batch, negative_batch], dim=0))

        pred = self.classifier_head(spatial_embedding)
        pred_type_unnorm = torch.cat([torch.zeros((len(pred), 1), device=self.device), pred], dim=1)   #TODO
        pred_type = F.softmax(pred_type_unnorm, dim=-1)
        pred_none = pred_type[:, 0]

        negtative_type = -torch.ones(len(negative_batch), device=self.device)
        target_type = torch.cat([mask_type, negtative_type], dim=0).to(torch.long) + 1 # 0 means no atom
        target_none = torch.cat([torch.zeros_like(mask_batch, device=self.device, dtype=torch.float),
                                 torch.ones_like(negative_batch, device=self.device, dtype=torch.float)], dim=0,) #TODO
        # target_type = F.one_hot(target_type, num_classes=len(atom_vocab)+2)
        # target_none = target_type[:, 0]


        # Frontier Prediction(based on atom bond?)
        old_edge_num = torch_scatter.scatter_add(src=torch.ones(ligand_graph.num_edge, device=self.device),
                                                 index=ligand_graph.edge_list[:,0],
                                                 dim_size=ligand_graph.num_node)
        old_edge_num = old_edge_num[~mask_node_index]
        new_edge_num = torch_scatter.scatter_add(src=torch.ones(ligand_graph_masked.num_edge, device=self.device),
                                                 index=ligand_graph_masked.edge_list[:,0],
                                                 dim_size=ligand_graph_masked.num_node)
        target_is_frontier = (old_edge_num > new_edge_num).to(torch.float)
        pred_is_frontier = self.frontier_model(context_embedding[complex_graph.is_ligand]).squeeze()

        # Pack Return
        pred = pred_type_unnorm, pred_none, pred_is_frontier
        target = target_type, target_none, target_is_frontier

        return pred, target

    def sample_step(self, ligand_graph, protein_graph, candidate_coord):
        candidate_num = candidate_coord.shape[0]
        complex_graph = self.getKNN(ligand_graph, protein_graph)
        context_embedding = self.complex_model(complex_graph, complex_graph.node_feature)["node_feature"]
        spatial_embedding = self.getSpatialEmbedding(context_coords=complex_graph.atom_coords,
                                                     context_batch=complex_graph.node2graph,
                                                     context_embedding=context_embedding,
                                                     predicted_coords=candidate_coord,
                                                     predicted_batch=torch.zeros(candidate_num, device=self.device)) #TODO
        pred = self.classifier_head(spatial_embedding)
        pred_type_unnorm = torch.cat([torch.zeros((len(pred), 1), device=self.device), pred], dim=1)  # TODO
        assert (pred_type_unnorm > 0).sum() > 0 # Must be some point with energy > 0
        pred_type_prob = F.softmax(pred_type_unnorm.reshape(-1))
        m = D.Categorical(pred_type_prob)
        flatten_index = m.sample()

        index = flatten_index / (len(atom_vocab) + 2)
        atom_coord = candidate_coord[index]
        atom_type = flatten_index % (len(atom_vocab) + 2) - 2

        return atom_coord, atom_type

    def init_MH_sample_step(self, sampling_model, initial_data):
        import pyro
        from pyro.infer import MCMC, NUTS
        nuts_kernel = NUTS(sampling_model)
        mcmc = MCMC(nuts_kernel, num_samples=500)
        mcmc.run(initial_data)
        samples = mcmc.get_samples()

    def evaluate(self, pred, target):
        metric = {}
        pred_type, pred_none, pred_is_frontier = pred
        target_type, target_none, target_is_frontier = target

        loss_BCE = F.binary_cross_entropy(pred_none, target_none)
        loss_CAT = F.cross_entropy(pred_type, target_type)
        loss_Frontier = F.binary_cross_entropy_with_logits(pred_is_frontier, target_is_frontier)


        metric["BCE Loss"] = loss_BCE
        metric["CAT Loss"] = loss_CAT
        metric["Frontier Classification Loss"] = loss_Frontier

        return metric
