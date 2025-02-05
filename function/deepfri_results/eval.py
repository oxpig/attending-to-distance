import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torcheval.metrics.functional import binary_auprc
from sklearn.metrics import PrecisionRecallDisplay


datas = [
    'DeepFRI-PDB_NoGraphConv_3x128_ca_10A_molecular_function_results.pckl',
    'DeepFRI-PDB_MultiGraphConv_3x512_ca_10A_molecular_function_results.pckl',
]

def auprc_loss(x):
    loss = binary_auprc
    display = PrecisionRecallDisplay.from_predictions(
        x['Y_true'].flatten(), x['Y_pred'].flatten(), name="Test", plot_chance_level=True
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    print('Max F1:')
    print(max(2/(1/display.precision + 1/display.recall)))
    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x_f = np.linspace(0.01, 1.0)
        y_f = f_score * x_f / (2 * x_f - f_score)
        plt.plot(x_f[y_f >= 0], y_f[y_f >= 0], color='gray', alpha=0.2)
        plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.85, y_f[45] + 0.02),
                     # fontsize=18, color='gray', alpha=0.7)
                    color='gray', alpha=0.7)
    print('AUPRC:')
    print(
        loss(
            torch.Tensor(x['Y_pred']).flatten(),
            torch.Tensor(x['Y_true']).flatten(),
        )
    )

for r_file in datas:
    print(r_file)
    with open(r_file, 'rb') as f:
        x = pickle.load(f)

    auprc_loss(x)
