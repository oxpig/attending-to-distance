import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_auprc, binary_f1_score
from const import BATCH_SIZE, NUM_EPOCHS


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Evaluator:
    def __init__(self, model=None, prot_data=None, results_path=None, transformer=False):
        if model is not None and not transformer:
            self._run_model(model, prot_data, results_path)
        elif model is not None:
            self._run_transformer(model, prot_data, results_path)
        else:
            self._load_results(results_path)

    def _load_results(self, results_path):
        self.results = torch.load(results_path)

    def _run_model(self, model, prot_data, results_path):
        model.eval()
        labels = []
        preds = []
        val_dataloader = DataLoader(
            prot_data.test_iter,
            batch_size=BATCH_SIZE,
            collate_fn=prot_data.collate_fn
        )
        for src, label in val_dataloader:
            src = src.to(DEVICE)
            label = label.cpu().detach().numpy()
            pred = model(src)
            pred = torch.sigmoid(pred).cpu().detach().numpy()
            labels.append(label)
            preds.append(pred)

        self.results = {
            'Y_true': torch.Tensor(np.concatenate(labels, 0)),
            'Y_pred': torch.Tensor(np.concatenate(preds, 0)),
        }
        torch.save(self.results, results_path)

    def _run_transformer(self, model, prot_data, results_path):
        model.eval()
        labels = []
        preds = []
        val_dataloader = DataLoader(prot_data, batch_size=BATCH_SIZE)
        for data in val_dataloader:
            src = data['in_seq'].to(DEVICE).transpose(0,1)
            src_pos = data['coords'].to(DEVICE).transpose(0,1)
            mask = data['mask'].to(DEVICE)
            label = data['annot'].cpu().detach().numpy()

            pred = model(src, src_pos, mask)
            src = src.to(DEVICE)
            pred = torch.sigmoid(pred).cpu().detach().numpy()
            labels.append(label)
            preds.append(pred)

        self.results = {
            'Y_true': torch.Tensor(np.concatenate(labels, 0)),
            'Y_pred': torch.Tensor(np.concatenate(preds, 0)),
        }
        torch.save(self.results, results_path)

    def ce(self):
        loss_fn = torch.nn.BCELoss()

        return loss_fn(
            self.results['Y_pred'],
            self.results['Y_true'],
        )

    def ce_hist(self):
        loss = torch.nn.BCELoss(reduction='none')
        loss_dist = loss(
            self.results['Y_pred'],
            self.results['Y_true'],
        ).flatten()
        loss_dist = list(loss_dist.detach().cpu().numpy())
        loss_dist = [loss for loss in loss_dist if loss < 0.02]
        counts, bins = np.histogram(loss_dist)
        plt.stairs(counts, bins)
        # plt.show()

    def aurc(self):
        from sklearn.metrics import PrecisionRecallDisplay
        loss = binary_auprc(
            self.results['Y_pred'].flatten(),
            self.results['Y_true'].flatten(),
        )
        display = PrecisionRecallDisplay.from_predictions(
            self.results['Y_true'].detach().cpu().numpy().flatten(),
            self.results['Y_pred'].detach().cpu().numpy().flatten(),
            name="Transformer",
            plot_chance_level=True
        )
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        f_scores = np.linspace(0.1, 0.9, num=9)
        for f_score in f_scores:
            x = np.linspace(0.01, 1.0)
            y = f_score * x / (2 * x - f_score)
            plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.85, y[45] + 0.02),
                         # fontsize=18, color='gray', alpha=0.7)
                        color='gray', alpha=0.7)
        plt.savefig('results/test_pr.pdf')
        self.precision = display.precision
        self.recall = display.recall
        return loss

    def confusion(self):
        from sklearn.metrics import ConfusionMatrixDisplay
        display = ConfusionMatrixDisplay.from_predictions(
            self.results['Y_true'].detach().cpu().numpy().flatten(),
            (self.results['Y_pred'].detach().cpu().numpy().flatten() > 0.5).astype(int),
        )
        plt.savefig('results/test_confusion.pdf')

    def max_f1(self):
        return max(2/(1/self.precision + 1/self.recall))

    def f1_score(self):
        loss = binary_f1_score(
            self.results['Y_pred'].flatten(),
            self.results['Y_true'].flatten(),
        )
        return loss

    def run_all(self):
        print('CE')
        loss = self.ce()
        print(loss)
        self.confusion()
        print('AUPR')
        loss = self.aurc()
        print(loss)
        print('MAX F1')
        loss = self.max_f1()
        print(loss)
