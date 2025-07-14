import os
import random
import torch


valid_prots = []
train_prots = []

for shard in os.listdir('.'):
    if not shard.endswith('.pt'):
        continue
    shard_data = torch.load(shard)

    for prot in shard_data:
        prot_id = prot[0]
        r = random.random()
        if r < 0.01:
            valid_prots.append(prot_id)
        else:
            train_prots.append(prot_id)

with open('train.txt', 'w') as f:
    f.write('\n'.join(train_prots))

with open('valid.txt', 'w') as f:
    f.write('\n'.join(valid_prots))
