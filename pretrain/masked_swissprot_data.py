# Adapted from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py

import os
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R

import esm

from utils.utils import load_coords


random.seed(42)


class MaskedProtData(Dataset):
    def __init__(
        self,
        split,
        shuffle,
        mask=True,
        all_a=False,
        shards_dir='./data/shards',
    ):
        self.split = split
        self.shuffle = shuffle
        self.mask = mask
        self.all_a = all_a
        self.shards_dir = shards_dir
        self.max_l = 500
        self.aas = 'ACDEFGHIKLMNPQRSTVWY'
        alphabet = self.aas + 'X[]~_-'
        self.no_coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.char2tok = {char: i for i, char in enumerate(alphabet)}
        self.tok2char = {i: char for i, char in enumerate(alphabet)}
        self.get_data()

    def get_data(self):
        split_data = []
        with open(f'{self.shards_dir}/{self.split}.txt', 'r') as f:
            split_prots = set([prot.strip() for prot in f.readlines()])
        for f_path in os.listdir(self.shards_dir):
            if not f_path.endswith('.pt'):
                continue
            shard_data = torch.load(self.shards_dir + os.sep + f_path)
            for prot in shard_data:
                prot_id = prot[0]
                if prot_id in split_prots:
                    seq = prot[1]
                    if len(seq) <= self.max_l:
                        coords = prot[2][:,1,:] # Just CA
                        split_data.append([prot_id, seq, coords])

        self.iter = split_data
        self.seq_len = max(len(prot[1]) for prot in self.iter)
        if self.shuffle:
            random.shuffle(self.iter)

    def __len__(self):
        return len(self.iter)

    def __getitem__(self, item):
        prot, seq, coords = self.iter[item]
        if self.mask:
            in_seq, out_seq = self.random_aa(seq)
        else:
            in_seq, out_seq = seq, seq
        coords = self.augment_coords(coords)
        if self.all_a:
            in_seq = 'A'*len(in_seq)

        output = {
            "prot": prot,
            "in_seq": in_seq,
            "out_seq": out_seq,
            "coords": coords,
        }
        return output

    def random_aa(self, seq):
        in_seq = ''
        out_seq = ''

        for aa in seq:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    in_seq += '<mask>'

                # 10% randomly change token to random token
                elif prob < 0.9:
                    in_seq += random.choice(self.aas)

                # 10% randomly change token to current token
                else:
                    in_seq += aa

                out_seq += aa

            else:
                in_seq += aa
                # out_seq += '-'
                out_seq += '<pad>'

        return in_seq, out_seq

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
