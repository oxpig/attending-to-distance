import math
import random

import torch
import numpy as np

from metric import distance

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rand_pos(n_nodes, dim=3):
    s = 200
    pos = [[s*random.random() for _ in range(dim)] for _ in range(n_nodes)]
    return np.array(pos, dtype=np.float32)


class PtData:
    def __init__(self, exp=2, total_pts=10000, dim=3):
        self.exp = exp
        self.total_pts = total_pts
        self.dim = dim
        self.get_data()

    def get_data(self):
        # data = torch.load(self.data_path)
        splits = []
        n_nodes = 5
        split_nums = {
            'train': int(0.8*self.total_pts),
            'val': int(0.1*self.total_pts),
            'test': int(0.1*self.total_pts),
        }
        for split in split_nums:
            n_pts = split_nums[split]
            pts = []
            for i in range(n_pts):
                one_hot = np.array([[0,1] for _ in range(n_nodes)], dtype=np.float32)
                pos = rand_pos(n_nodes, dim=self.dim)
                label = np.array([
                    [distance(pos[i], pos[j], self.exp) for i in range(n_nodes)]
                    for j in range(n_nodes)]
                )
                pts.append([one_hot, pos, label])
            splits.append(pts)

        self.val_iter, self.test_iter, self.train_iter = splits
        random.shuffle(self.train_iter)


    # function to collate data samples into batch tensors
    def collate_fn(self, batch):
        src_batch, src_pos_batch, label_batch = [], [], []

        for src_sample, src_pos_sample, label_sample in batch:
            src_batch.append(src_sample)
            src_pos_batch.append(src_pos_sample)
            label_batch.append(label_sample)

        src_batch = torch.swapaxes(torch.tensor(src_batch, device=DEVICE),0,1)
        src_pos_batch = torch.swapaxes(torch.tensor(src_pos_batch, device=DEVICE),0,1)
        label_batch = torch.tensor(label_batch, device=DEVICE)
        return (
            src_batch,
            src_pos_batch,
            label_batch,
        )

