# Adapted from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py

import os
import math
import random
import pickle as pkl

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist, squareform
import lmdb

import esm

from utils.utils import load_coords


random.seed(42)


class ContactData(Dataset):
    def __init__(
        self,
        split,
        # shuffle,
        mask=True,
        all_a=False,
        data_dir='./data/contact/proteinnet',
    ):
        self.split = split
        # self.shuffle = shuffle
        self.mask = mask
        self.all_a = all_a
        self.data_dir = data_dir
        self.max_l = 1000
        # self.max_l = 500
        self.aas = 'ACDEFGHIKLMNPQRSTVWY'
        alphabet = self.aas + 'X[]~_-'
        self.no_coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.char2tok = {char: i for i, char in enumerate(alphabet)}
        self.tok2char = {i: char for i, char in enumerate(alphabet)}
        self.get_data()

    def get_data(self):
        split_data = []
        env = lmdb.open(self.data_dir + os.sep + f'proteinnet_{self.split}.lmdb', max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self._num_examples = pkl.loads(txn.get(b'num_examples'))
        self._env = env

    def __len__(self):
        # return min(1000, self._num_examples)
        return self._num_examples

    def __getitem__(self, item):
        with self._env.begin(write=False) as txn:
            data = pkl.loads(txn.get(str(item).encode()))
            if 'id' not in data:
                data['id'] = str(index)
        prot = data['id']
        seq = data['primary'][:self.max_l]
        coords = data['tertiary'][:self.max_l]
        valid_mask = data['valid_mask'][:self.max_l]
        contact_map = np.less(squareform(pdist(coords)), 8.0).astype(np.int64)
        y_inds, x_inds = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(y_inds - x_inds) < 6
        contact_map[invalid_mask] = -1

        coords = self.augment_coords(coords)
        if self.all_a:
            in_seq = 'A'*len(in_seq)

        output = {
            "prot": prot,
            "in_seq": seq,
            "coords": coords,
            "contact_map": contact_map,
        }
        return output

    def augment_coords(self, coords):
        # Recentre
        has_coords = np.any(coords[:,:3] != 0, 1) # All zeros means no pos
        com = coords[has_coords].mean(0) # Centre of mass
        coords[has_coords] = coords[has_coords] - com
        n_pts, _ = coords.shape
        new_coords = np.empty(coords.shape, dtype=coords.dtype)
        # Random rotation
        r = R.random()
        for i in range(n_pts):
            new_coords[i,:3] = r.apply(coords[i,:3])
        return new_coords
