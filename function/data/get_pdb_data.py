import os
from collections import defaultdict
import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import *
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1_extended as three_to_one


pdbl = PDBList()
parser = MMCIFParser()
aligner = PairwiseAligner()
no_coord = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
not_missing = np.array([0.0], dtype=np.float32)
obsolete = set(pdbl.get_all_obsolete())

data_dir = 'data/pdb_coords'
shard_size = 1000
max_names = 10**10 # (can be small for testing)
shard_idx = 0
data = {}

pdbs = defaultdict(list)

ont = 'GO'
seq_data = {}

with open(f"data/nrPDB-{ont}_2019.06.18_sequences.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq_data[record.id] = record.seq

latest = 0
complete = set()

for i in range(latest):
    old_data = np.load(f'{data_dir}/CA_coords_{ont}_{i}.npz')

    for file in old_data.files:
        complete.add(file)

    old_data.close()


shard_idx += latest
names = sorted(list(seq_data.keys()))

for name in names[:max_names]:
    if name in complete:
        continue
    pdb, chain = name.split('-')
    if pdb not in obsolete:
        pdbs[pdb].append(chain)

failed = []

for pdb in pdbs:
    for _ in range(5):
        # This sometimes fails randomly so, as a hack, we try it five times
        pdbl.retrieve_pdb_file(pdb.lower(), pdir=data_dir)
    pdb_path = f"{data_dir}/{pdb.lower()}.cif"
    try:
        structure = parser.get_structure(pdb, pdb_path)
    except:
        failed.append(pdb)
        os.remove(pdb_path)
        continue
    model = structure[0]
    for chain_id in pdbs[pdb]:
        ca_coords = []
        name = f'{pdb}-{chain_id}'
        chain = model[chain_id]
        # PDB structures are frequently missing residues so we align to the
        # given sequence and add coordinates when they exist
        seq = seq_data[name]
        pdb_residues = [res for res in chain if res.get_resname() in three_to_one]
        pdb_seq = ''.join([three_to_one[res.get_resname()] for res in pdb_residues])
        if len(pdb_seq) == 0:
            ca_coords = [no_coord for _ in seq]
        else:
            alignment = next(aligner.align(seq, pdb_seq))
            seq_idxs, pdb_idxs = alignment.indices

            for i in range(len(seq_idxs)):
                seq_idx = seq_idxs[i]
                pdb_idx = pdb_idxs[i]
                if seq_idx == -1:
                    # There is a gap in the original sequence because
                    # something weird was in the PDB chain
                    continue
                if pdb_idx == -1 or 'CA' not in pdb_residues[pdb_idx]:
                    # The coordinate is missing
                    ca_coords.append(no_coord)
                else:
                    # We have the coordinate
                    residue = pdb_residues[pdb_idx]
                    ca_coords.append(np.concatenate((residue['CA'].get_coord(), not_missing)))

        ca_arr = np.array(ca_coords)
        data[name] = ca_arr
        if len(data) == shard_size:
            with open(f'{data_dir}/CA_coords_{ont}_{shard_idx}.npz', 'wb') as f:
                np.savez(f, **data)
            data = {}
            shard_idx += 1
    os.remove(pdb_path)

# Leftovers from last shard
if len(data) > 0:
    with open(f'{data_dir}/CA_coords_{ont}_{shard_idx}.npz', 'wb') as f:
        np.savez(f, **data)

with open(f'{data_dir}/failed.txt', 'w') as f:
    f.write('\n'.join(failed))
