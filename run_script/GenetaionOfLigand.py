import yaml
import numpy as np
import random
from easydict import EasyDict
import torch
from torch import optim

from torchdrug import data, models, core

from dataset.crossdocked import CrossDockedSubset
from task.generation_3d import LigandGeneration

# def test():
#     dataset = CrossDockedSubset("~/dataset/crossdocked_subset/crossdocked_subset/1433B_HUMAN_1_240_pep_0")
#     dataloader = data.DataLoader(dataset, 2)
#     node_feature_dim_complex = dataset.node_feature_dim_protein + dataset.node_feature_dim_ligand
#     complex_model = models.SchNet(input_dim=node_feature_dim_complex, hidden_dims=[128, 128, 128])
#     spatial_model = models.SchNet(input_dim=complex_model.output_dim, hidden_dims=[128, 128])
#     task = LigandGeneration(complex_model, spatial_model)
#     for i, batch in enumerate(dataloader):
#         loss = task(batch)
#         print(loss)

def train():

    cfg = EasyDict()
    cfg.optimizer = EasyDict()
    cfg.train = EasyDict()
    cfg.train.epoch = 20

    # config_path = "~/GoL.yaml"
    # with open(config_path, 'r') as f:
    #     cfg = yaml.safe_load(f)

    # set random seed
    #torch.autograd.set_detect_anomaly(True)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')


    dataset = CrossDockedSubset(
        "~/dataset/crossdocked_subset/crossdocked_subset/")
    node_feature_dim_complex = dataset.node_feature_dim_protein + dataset.node_feature_dim_ligand
    complex_model = models.SchNet(input_dim=node_feature_dim_complex, hidden_dims=[128, 128, 128])
    spatial_model = models.SchNet(input_dim=complex_model.output_dim, hidden_dims=[128, 128])
    task = LigandGeneration(complex_model, spatial_model)
    optimizer = optim.Adam(task.parameters(), **cfg.optimizer)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    solver = core.Engine(task, dataset, None, None, optimizer, gpus=[0])
    solver.train(num_epoch=cfg.train.epoch)


if __name__ == "__main__":

    train()