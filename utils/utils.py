import torch
def getIndex(num_protein_nodes, num_ligand_nodes):
    index_protein_list = []
    index_ligand_list = []
    for batch_id in range(len(num_protein_nodes)):
        cum_protein = num_protein_nodes[:batch_id].sum()
        index_protein_tmp = torch.arange(cum_protein, cum_protein+num_protein_nodes[batch_id])  # (M)
        index_protein_tmp = index_protein_tmp.unsqueeze(-1).expand(
            num_protein_nodes[batch_id], num_ligand_nodes[batch_id]
        ).reshape(-1)  # (MxN)
        index_protein_list.append(index_protein_tmp)

        cum_ligand = num_ligand_nodes[:batch_id].sum()
        index_ligand_tmp = torch.arange(cum_ligand, cum_ligand+num_ligand_nodes[batch_id])  # (N)
        index_ligand_tmp = index_ligand_tmp.unsqueeze(0).expand(
            num_protein_nodes[batch_id], num_ligand_nodes[batch_id]
        ).reshape(-1)
        index_ligand_list.append(index_ligand_tmp)

    index_protein = torch.cat(index_protein_list)
    index_ligand = torch.cat(index_ligand_list)

    return index_protein, index_ligand

def get_bigraph_distance(coords_x, coords_y, index_x, index_y):
    return (coords_x[index_x] - coords_y[index_y]).norm(dim=-1)