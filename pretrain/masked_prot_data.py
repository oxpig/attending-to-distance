# Adapted from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py

import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.deep_fri_utils import load_FASTA
from utils.utils import load_coords


random.seed(42)


class MaskedProtData(Dataset):
    def __init__(
        self,
        split,
        shuffle,
        mask=True,
        all_a=False,
        data_prefix='../function/data/pretrain/pretrain',
        fasta_path='../function/data/nrPDB-GO_2019.06.18_sequences.fasta',
        coords_path='../function/data/pdb_coords/',
    ):
        self.split = split
        self.shuffle = shuffle
        self.mask = mask
        self.all_a = all_a
        self.data_prefix = data_prefix
        self.fasta_path = fasta_path
        self.coords_path = coords_path
        self.aas = 'ACDEFGHIKLMNPQRSTVWY'
        alphabet = self.aas + 'X[]~_-'
        self.no_coord = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.char2tok = {char: i for i, char in enumerate(alphabet)}
        self.tok2char = {i: char for i, char in enumerate(alphabet)}
        self.get_data()

    def get_data(self):
        # TODO: the alignments aren't great for the coords

        prots, seqs = load_FASTA(self.fasta_path)
        prot2coords = load_coords(self.coords_path, add_class=False)
        with open(f'{self.data_prefix}_{self.split}.txt', 'r') as f:
            split_prots = set([prot.strip() for prot in f.readlines()])
        n_prots = len(prots)
        split_data = []
        for i in range(n_prots):
            prot = prots[i]
            if prot not in prot2coords:
                # We couldn't parse a few pdb entries
                continue
            if prot not in split_prots:
                continue
            seq = seqs[i]
            prot_data = [
                prot,
                seq,
                prot2coords[prot],
            ]
            split_data.append(prot_data)

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
        num_pads = self.seq_len - len(in_seq)
        padding = '-' * num_pads
        if self.all_a:
            in_seq = 'A'*len(in_seq)
        in_seq = '[' + in_seq + ']' + padding
        out_seq = '-' + out_seq + '-' + padding
        coords = np.concatenate((self.no_coord, coords, self.no_coord), 0)
        pad_coords = np.concatenate((coords, np.zeros((num_pads, 4))), 0, dtype=np.float32)
        mask = [char == '-' for char in in_seq]

        output = {
            "prot": prot,
            "in_seq": [self.char2tok[char] for char in in_seq],
            "out_seq": [self.char2tok[char] for char in out_seq],
            "mask": mask,
            "coords": pad_coords,
        }

        return {
            key: torch.tensor(value) if key != "prot" else value
            for key, value in output.items()
        }

    def random_aa(self, seq):
        in_seq = ''
        out_seq = ''

        for aa in seq:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    in_seq += '~'

                # 10% randomly change token to random token
                elif prob < 0.9:
                    in_seq += random.choice(self.aas)

                # 10% randomly change token to current token
                else:
                    in_seq += aa

                out_seq += aa

            else:
                in_seq += aa
                out_seq += '-'

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
            new_coords[i,3] = coords[i,3] # Coord missing flag
        return new_coords
