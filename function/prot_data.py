import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from utils.deep_fri_utils import load_GO_annot, load_FASTA, seq2onehot
from utils.utils import list_prots, load_coords


random.seed(42)

class ProtData:
    def __init__(
        self,
        ont,
        data_prefix='data/nrPDB-GO_2019.06.18',
        fasta_path='data/nrPDB-GO_2019.06.18_sequences.fasta',
        annot_path='data/nrPDB-GO_2019.06.18_annot.tsv',
        coords_path='data/pdb_coords/',
        emb_path='../pretrain/data/bert_coords_new',
    ):
        self.ont = ont
        self.data_prefix = data_prefix
        self.fasta_path = fasta_path
        self.annot_path = annot_path
        self.coords_path = coords_path
        self.emb_path = emb_path
        self.get_data()

    def get_data(self):
        split_names = ['valid', 'test', 'train']
        prot2annot = load_GO_annot(self.annot_path)[0]
        prots, seqs = load_FASTA(self.fasta_path)
        prots_with_embs = list_prots(self.coords_path)
        splits = []
        for split in split_names:
            with open(f'{self.data_prefix}_{split}.txt', 'r') as f:
                split_prots = set([prot.strip() for prot in f.readlines()])
            n_prots = len(prots)
            split_data = []
            for i in range(n_prots):
                prot = prots[i]
                if prot not in prots_with_embs:
                    # We couldn't parse a few pdb entries
                    continue
                if prot not in split_prots:
                    continue
                prot_data = []
                prot_data.append(prot)
                prot_data.append(prot2annot[prot][self.ont])
                split_data.append(prot_data)
            splits.append(split_data)

        self.val_iter, self.test_iter, self.train_iter = splits
        random.shuffle(self.train_iter)

    # function to collate pretrained embeddings into batch tensors
    def collate_fn(self, batch):
        emb_batch, label_batch = [], []

        for prot_sample, label_sample in batch:
            emb_batch.append(torch.load(f'{self.emb_path}/{prot_sample}.pt'))
            label_batch.append(label_sample)

        emb_batch = torch.stack(emb_batch)
        label_batch = torch.Tensor(label_batch)
        return (
            emb_batch,
            label_batch,
        )
