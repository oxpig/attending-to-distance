import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

splits = ['train', 'val']
for split in splits:
    df = pd.read_csv(f'{split}_loss.csv')
    plt.plot(
        df['Step'],
        np.exp(df[f'with-coords-esm1-tiny - {split}_loss']),
        label=f'With coords ({split})',
    )
    plt.plot(
        df['Step'],
        np.exp(df[f'no-coords-esm1-tiny - {split}_loss']),
        label=f'No coords ({split})',
    )
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Train perplexity', fontsize=16)
    plt.legend()
    plt.tight_layout()

plt.savefig(f'pretraining_loss.pdf')
