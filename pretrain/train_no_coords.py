import torch

from utils.training import train_model
from models.model import BERTCoords
from masked_prot_data import MaskedProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 768
num_heads = 12
num_layers=6

train_data = MaskedProtData('train', shuffle=True)
val_data = MaskedProtData('valid', shuffle=False)
vocab_size = len(train_data.char2tok.keys())

model = BERTCoords(
    num_encoder_layers=num_layers,
    emb_size=d_model,
    nhead=num_heads,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    dropout=0.0,
    add_coords=False,
)
model = model.to(DEVICE)


with torch.backends.cuda.sdp_kernel(enable_math=False):
    train_model(model, train_data, val_data, 'bert_no_coords')
