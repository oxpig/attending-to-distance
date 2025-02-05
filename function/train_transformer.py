import torch

from utils.transformer_training import train_model
from models.bert_coords import BERTCoords
from models.transformer import TransformerModel
from masked_prot_data import MaskedProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_transformer(model_path, add_coords=True):
    d_model = 768
    num_heads = 12
    num_layers = 6

    vocab_size = 26

    model = BERTCoords(
        num_encoder_layers=num_layers,
        emb_size=d_model,
        nhead=num_heads,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dropout=0,
        add_coords=add_coords,
    )
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model


def make_model(transformer, input_dim=768, num_onts=489):
    return TransformerModel(
        input_dim=768,
        num_onts=489,
        device=DEVICE,
        transformer=transformer,
    ).to(DEVICE)

no_coords_transformer = load_transformer('../pretrain/weights/best_weights_bert_no_coords.pt', add_coords=False)
no_coords_model = make_model(no_coords_transformer)
coords_transformer = load_transformer('../pretrain/weights/best_weights_bert_coords.pt')
coords_model = make_model(coords_transformer)

train_data = MaskedProtData('train', shuffle=True, ont='mf', mask=False)
val_data = MaskedProtData('valid', shuffle=False, ont='mf', mask=False)

train_model(no_coords_model, train_data, val_data, model_name='no_coords_transformer')
train_model(coords_model, train_data, val_data, model_name='coords_transformer')
