import torch

from utils.contact_training import train_contact
from models.esm_hybrid_model import BERTCoords
from contact_data import ContactData

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = ContactData('train')
val_data = ContactData('valid')
# val_data = ContactData('test')

model = BERTCoords(
    dropout=0.0,
    # add_coords=False,
)
model.load_state_dict(torch.load('weights/best_weights_bert_coords_swissprot.pt'))
# model.load_state_dict(torch.load('weights/best_weights_bert_no_coords_swissprot.pt'))

for i in range(6):
    model.encoder.layers[i]._requires_grad = False

model = model.to(DEVICE)


with torch.backends.cuda.sdp_kernel(enable_math=False):
    train_contact(model, train_data, val_data, 'bert_coords_contact')
    # train_contact(model, train_data, val_data, 'bert_no_coords_contact')
