import os
import numpy as np


def list_prots(coords_path):
    prots = set()
    for file in os.listdir(coords_path):
        filename = os.fsdecode(file)
        if filename.startswith('CA_coords_GO') and filename.endswith('.npz'):
            data = np.load(f'{coords_path}/{filename}')
            for prot in data.files:
                prots.add(prot)
            data.close()
    return prots


def load_coords(coords_path, add_class=True):
    no_coord = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    prot2coords = {}
    for file in os.listdir(coords_path):
        filename = os.fsdecode(file)
        if filename.startswith('CA_coords_GO') and filename.endswith('.npz'):
            data = np.load(f'{coords_path}/{filename}')
            for prot in data.files:
                coords = data[prot]
                if add_class:
                    coords = np.concatenate((no_coord, coords), 0)
                prot2coords[prot] = coords
            data.close()
    return prot2coords
