import math
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from scipy.constants import physical_constants

from const import BATCH_SIZE, NUM_EPOCHS
from .noam_opt import get_std_opt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, loss_fn, val_data):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
 
    for data in val_dataloader:
        src = data['in_seq'].to(DEVICE).transpose(0,1)
        src_pos = data['coords'].to(DEVICE).transpose(0,1)
        tgt = data['out_seq'].to(DEVICE).transpose(0,1)
        mask = data['mask'].to(DEVICE)
 
        logits = model(src, src_pos, mask)
 
        loss = loss_fn(
            logits.permute(1,2,0),
            tgt.permute(1,0)
        )
        losses += loss.item()
 
    return losses / len(val_dataloader)


def train_epoch(model, optimizer, train_data, loss_fn):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

    for data in train_dataloader:
        src = data['in_seq'].to(DEVICE).transpose(0,1)
        src_pos = data['coords'].to(DEVICE).transpose(0,1)
        tgt = data['out_seq'].to(DEVICE).transpose(0,1)
        mask = data['mask'].to(DEVICE)

        logits = model(src, src_pos, mask)

        optimizer.zero_grad()

        loss = loss_fn(
            logits.permute(1,2,0),
            tgt.permute(1,0)
        )
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def train_model(model, train_data, val_data, model_name='bert'):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_data.char2tok['-'])
    optimizer = get_std_opt(model.parameters(), model.emb_size)

    best_loss = math.inf
    weights_path = f'weights/best_weights_{model_name}.pt'

    for epoch in range(1, NUM_EPOCHS+1):
        print(f"Training Epoch {epoch}")
        start_time = timer()
        train_loss = train_epoch(model, optimizer, train_data, loss_fn)
        end_time = timer()
        val_loss = evaluate(model, loss_fn, val_data)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if val_loss < best_loss:
            print('Validation loss is lower than best loss, saving...')
            best_loss = val_loss
            torch.save(model.state_dict(), weights_path)
