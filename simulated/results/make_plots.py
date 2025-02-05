import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_exp_loss():
    plt.clf()
    df = pd.read_csv('data/exp_loss.csv')
    plt.bar(df['exp'], df['val_loss'], width=0.3)
    plt.title('Accuracy at determining $exp({-d^{p}})$', fontsize=16)
    plt.xlabel('Exponent (p)', fontsize=16)
    plt.ylabel('Validation loss', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/exp_loss.pdf')


def plot_emb_dim_range(dims=4):
    plt.clf()
    for dim in range(dims):
        dim += 1
        df = pd.read_csv(f'data/emb_dim_loss_{dim}.csv')
        plt.plot(df['emb_size'], df['val_loss'], label=f'Spatial dimension = {dim}')
    plt.title('Accuracy depending on head dimension', fontsize=16)
    plt.xlabel('Head dimension', fontsize=16)
    plt.ylabel('Validation loss', fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/emb_dim_loss.pdf')


def plot_augment():
    plt.clf()
    df = pd.read_csv('data/epoch_losses_simple.csv')
    df_a = pd.read_csv('data/epoch_losses_augmented.csv')
    window = 500
    plt.plot(df['epoch'], df.rolling(window).mean()['val_loss'], label='Raw validation')
    plt.plot(df['epoch'], df.rolling(window).mean()['train_loss'], label='Raw train')
    plt.plot(df_a['epoch'], df_a.rolling(window).mean()['val_loss'], label='Augmented validation')
    plt.plot(df_a['epoch'], df_a.rolling(window).mean()['train_loss'], label='Augmented train')
    plt.title('Rolling average loss per epoch')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('plots/augment_losses.pdf')

plot_exp_loss()
# plot_emb_dim_range()
# plot_augment()
