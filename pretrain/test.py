from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.model import BERTCoords
from masked_prot_data import MaskedProtData

from const import BATCH_SIZE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 768
num_heads = 12
num_layers=6

train_data = MaskedProtData('train', shuffle=True)
val_data = MaskedProtData('valid', shuffle=False)
vocab_size = len(train_data.char2tok.keys())


def seq_recovery(model, val_data):
    model.eval()
    corrects = defaultdict(int)
    totals = defaultdict(int)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
 
    for data in val_dataloader:
        src = data['in_seq'].to(DEVICE).transpose(0,1)
        src_pos = data['coords'].to(DEVICE).transpose(0,1)
        tgt = data['out_seq'].to(DEVICE).transpose(0,1)
        mask = data['mask'].to(DEVICE)
 
        logits = model(src, src_pos, mask)
        mask = tgt != 25
        tgt = tgt[mask]
        pred = logits.max(2).indices[mask]
        for i in range(len(tgt)):
            tgt_tok = int(tgt[i])
            pred_tok = int(pred[i])
            totals[tgt_tok] += 1
            if tgt_tok == pred_tok:
                corrects[tgt_tok] += 1
 
    return {train_data.tok2char[tok]: [corrects[tok], totals[tok]] for tok in totals}

coords_model = BERTCoords(
    num_encoder_layers=num_layers,
    emb_size=d_model,
    nhead=num_heads,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    dropout=0,
).to(DEVICE)

no_coords_model = BERTCoords(
    num_encoder_layers=num_layers,
    emb_size=d_model,
    nhead=num_heads,
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    dropout=0,
    add_coords=False
).to(DEVICE)

def load_weights(model, model_path='weights/best_weights.pt'):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))


load_weights(coords_model, model_path="weights/best_weights_bert_coords.pt")
load_weights(no_coords_model, model_path="weights/best_weights_bert_no_coords.pt")

coords_loss = seq_recovery(coords_model, val_data)
no_coords_loss = seq_recovery(no_coords_model, val_data)
toks = []
diffs = []
coords_accs = []
no_coords_accs = []
for tok in coords_loss:
    if tok == 'X':
        # There are a few unknown tokens that were not filtered
        continue
    toks.append(tok)
    print(tok)
    coords_acc = coords_loss[tok][0] / coords_loss[tok][1]
    no_coords_acc = no_coords_loss[tok][0] / no_coords_loss[tok][1]
    print(coords_acc)
    print(no_coords_acc)
    diffs.append(coords_acc - no_coords_acc)
    coords_accs.append(coords_acc)
    no_coords_accs.append(no_coords_acc)


toks = [tok for _, tok in sorted(zip(coords_accs, toks))]
no_coords_accs = [acc for _, acc in sorted(zip(coords_accs, no_coords_accs))]
coords_accs = [acc for _, acc in sorted(zip(coords_accs, coords_accs))]

df = pd.DataFrame({
    'aa': toks,
    'coords_rate': coords_accs,
    'no_coords_rate': no_coords_accs,
})
df.to_csv('results/aa_recoveries.csv', index=False)

