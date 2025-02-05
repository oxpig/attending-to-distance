import torch
from torch.utils.data import DataLoader

from const import BATCH_SIZE
from models.model import BERTCoords
from masked_prot_data import MaskedProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 768
num_heads = 12
num_layers=6

train_data = MaskedProtData('train', shuffle=True, mask=False)
val_data = MaskedProtData('valid', shuffle=False, mask=False)
vocab_size = len(train_data.char2tok.keys())

model = BERTCoords(
    num_encoder_layers=num_layers,
    emb_size=d_model,
    nhead=num_heads,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    dropout=0.0,
)
model = model.to(DEVICE)

def load_weights(model, model_path='weights/best_weights.pt'):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))


load_weights(model, model_path="weights/best_weights_bert_coords.pt")

model.eval()
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
# dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
data_dir = "data/bert_coords_new"

for batch_idx, data in enumerate(dataloader):
    print(
        f"Processing {batch_idx + 1} of {len(dataloader)} batches"
    )
    src = data['in_seq'].to(DEVICE).transpose(0,1)
    src_pos = data['coords'].to(DEVICE).transpose(0,1)
    tgt = data['out_seq'].to(DEVICE).transpose(0,1)
    mask = data['mask'].to(DEVICE)
    prots = data['prot']

    emb = model(src, src_pos, mask, return_emb=True)
    for i, prot in enumerate(prots):
        seq_len = mask[i].int().sum().item()
        seq_embs = emb[:seq_len,i,:].clone()
        torch.save(seq_embs.mean(0), f'{data_dir}/{prot}.pt')
