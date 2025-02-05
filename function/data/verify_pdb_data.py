import numpy as np
from Bio import SeqIO

seq_data = {}

with open("data/nrPDB-GO_2019.06.18_sequences.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq_data[record.id] = record.seq

data_dir = 'data/pdb_coords'

good = True

ct = 0

names = set()

for i in range(37):
    data = np.load(f'{data_dir}/CA_coords_GO_{i}.npz')

    for file in data.files:
        names.add(file)
        ct += 1
        c_len = len(data[file])
        s_len = len(seq_data[file])
        if c_len != s_len:
            print(file)
            print(c_len)
            print(s_len)
            good = False

    data.close()

print(f'Collectively, the files contain {ct} structures')
print(f'Collectively, the files contain {len(names)} unique structures')
print(f'The fasta contains {len(seq_data)} sequences')

if good:
    print('All the lengths look good!')
