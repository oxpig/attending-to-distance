import torch

from utils.evaluation import Evaluator
from models.esm_mlp import MLPModel
from prot_data import ProtData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_model(input_dim=768, num_onts=489):
    return MLPModel(
        input_dim=768,
        num_onts=489,
        device=DEVICE,
    ).to(DEVICE)

def load_weights(model, model_path='weights/best_weights_esm.pt'):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

# no_coords_model = make_model()
# coords_model = make_model()

# no_coords_data = ProtData('mf', emb_path='../pretrain/data/bert_no_coords_new')
# coords_data = ProtData('mf', emb_path='../pretrain/data/bert_coords_new')

# load_weights(esm_model, model_path="weights/best_weights_esm.pt")
# load_weights(no_coords_model, model_path="weights/best_weights_no_coords.pt")
# load_weights(coords_model, model_path="weights/best_weights_coords_tiny.pt")

# evaluator = Evaluator(no_coords_model, no_coords_data, 'results/no_coords_results.pt')
evaluator = Evaluator(None, None, 'results/no_coords_results.pt')
print('No coords results:')
evaluator.run_all()
# evaluator = Evaluator(coords_model, coords_data, 'results/coords_results.pt')
evaluator = Evaluator(None, None, 'results/coords_results.pt')
print('Coords results:')
evaluator.run_all()
