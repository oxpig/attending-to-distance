import os

import torch
import esm
import numpy

from util import load_coords

entries = []
idx = 0

for pdb_file in os.listdir():
    # pdb_file = 'AF-A4QL21-F1-model_v4.pdb.gz'
    if not pdb_file.endswith('.pdb.gz'):
        continue
    name = pdb_file.split('.')[0]

    coords, seq = load_coords(pdb_file, 'A')
    entries.append([name, seq, coords])
    if len(entries) == 10000:
        print(f'Saving shard {idx}')
        torch.save(entries, f'prots_{idx}.pt')
        entries = []
        idx += 1

torch.save(entries, f'prots_{idx}.pt')
