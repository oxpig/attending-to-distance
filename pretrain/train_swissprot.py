import torch

from utils.training import train_model
from models.esm_hybrid_model import BERTCoords
from masked_swissprot_data import MaskedProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = MaskedProtData('train', shuffle=True)
val_data = MaskedProtData('valid', shuffle=False)

model = BERTCoords(
    dropout=0.0,
)
model = model.to(DEVICE)


with torch.backends.cuda.sdp_kernel(enable_math=False):
    train_model(model, train_data, val_data, 'bert_coords_swissprot')
