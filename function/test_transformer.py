import torch

from utils.evaluation import Evaluator
from models.bert_coords import BERTCoords
from models.transformer import TransformerModel
from masked_prot_data import MaskedProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_transformer(add_coords=True):
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
    return model


def make_model(transformer, model_path, input_dim=768, num_onts=489):
    model = TransformerModel(
        input_dim=768,
        num_onts=489,
        device=DEVICE,
        transformer=transformer,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model

# no_coords_transformer = load_transformer(add_coords=False)
# no_coords_model = make_model(no_coords_transformer, 'weights/best_weights_no_coords_transformer.pt')
# coords_transformer = load_transformer()
# coords_model = make_model(coords_transformer, 'weights/best_weights_coords_transformer.pt')

# test_data = MaskedProtData('test', shuffle=False, ont='mf', mask=False)

# evaluator = Evaluator(no_coords_model, test_data, 'results/no_coords_transformer_results.pt', transformer=True)
evaluator = Evaluator(None, None, 'results/no_coords_transformer_results.pt', transformer=True)
print('No coords results:')
evaluator.run_all()
# evaluator = Evaluator(coords_model, test_data, 'results/coords_transformer_results.pt', transformer=True)
evaluator = Evaluator(None, None, 'results/coords_transformer_results.pt', transformer=True)
print('Coords results:')
evaluator.run_all()
