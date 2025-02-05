import torch

from utils.training import train_model
from models.esm_mlp import MLPModel
from prot_data import ProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_model(input_dim=768, num_onts=489):
    return MLPModel(
        input_dim=768,
        num_onts=489,
        device=DEVICE,
    ).to(DEVICE)

no_coords_model = make_model()
coords_model = make_model()

no_coords_data = ProtData('mf', emb_path='../pretrain/data/bert_no_coords_new')
coords_data = ProtData('mf', emb_path='../pretrain/data/bert_coords_new')

train_model(no_coords_model, no_coords_data, model_name='no_coords')
train_model(coords_model, coords_data, model_name='coords_tiny')
