import torch
from new_transformer import TransformerEncoder, TransformerEncoderLayer

import pandas as pd

from const import d_model, num_heads, num_layers, dim_feedforward
from model import StructLangReg
from pt_data import PtData
from training import train_model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype=torch.float


def get_model(
        dropout=0.0,
        d_model=d_model,
        num_heads=num_heads,
        dim=3,
        activation='relu'
    ):
    model = StructLangReg(
        num_encoder_layers=num_layers,
        emb_size=d_model,
        nhead=num_heads,
        src_vocab_size=dim,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    return model.to(DEVICE)


def train_exp_range():
    exps = []
    val_losses = []
    for i in range(1, 9):
        exp = i/2
        print(f'Training for exp={exp}')
        model = get_model()
        pt_data = PtData(exp)

        best_val_loss = train_model(model, pt_data, save_best=False, augment=True)
        exps.append(exp)
        val_losses.append(best_val_loss)

    df = pd.DataFrame({'exp': exps, 'val_loss': val_losses})
    df.to_csv(f'results/data/exp_loss.csv', index=False)


def train_emb_dim_range(dim=3):
    pt_data = PtData(dim=dim)
    emb_sizes = []
    val_losses = []
    num_trials = 1
    for i in range(8):
        emb_dim = i+1
        best_val_losses = []
        for j in range(num_trials):
            print(f'Training for emb_dim={emb_dim}')
            model = get_model(d_model=emb_dim*8, num_heads=1, dim=dim)

            best_val_loss = train_model(model, pt_data, save_best=False, augment=True)
            best_val_losses.append(best_val_loss)
        emb_sizes.append(emb_dim)
        val_losses.append(sum(best_val_losses)/num_trials)

    df = pd.DataFrame({'emb_size': emb_sizes, 'val_loss': val_losses})
    df.to_csv(f'results/data/emb_dim_loss_{dim}.csv', index=False)


def train_normal(dim=3, plot_name=None, augment=False, n_pts=10000):
    model = get_model(dim=dim)
    pt_data = PtData(dim=dim, total_pts=n_pts)
    if plot_name is not None:
        plots_path = f'results/plots/{plot_name}.pdf'
    else:
        plots_path = None

    best_val_loss = train_model(model, pt_data, plots_path=plots_path, augment=augment)


def train_augment_data():
    train_normal(dim=3, n_pts=100, plot_name='epoch_losses_simple')
    train_normal(dim=3, n_pts=100, plot_name='epoch_losses_augmented', augment=True)


train_augment_data()
train_exp_range()
for dim in range(4):
    train_emb_dim_range(dim=dim+1)
