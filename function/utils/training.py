import math
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from torcheval.metrics.functional import binary_auprc, binary_f1_score

from const import BATCH_SIZE, NUM_EPOCHS


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, loss_fn, prot_data):
    model.eval()
    losses = 0
 
    val_dataloader = DataLoader(
        prot_data.val_iter,
        batch_size=BATCH_SIZE,
        collate_fn=prot_data.collate_fn
    )
 
    for src, label in val_dataloader:
        src = src.to(DEVICE)
        label = label.to(DEVICE)
 
        pred = model(src)
 
        loss = loss_fn(pred, label)
        losses += loss.item()
 
    return losses / len(val_dataloader)


def evaluate_aupr(model, prot_data):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(
        prot_data.val_iter,
        batch_size=BATCH_SIZE,
        collate_fn=prot_data.collate_fn
    )

    for src, label in val_dataloader:
        src = src.to(DEVICE)
        label = label.to(DEVICE)

        pred = model(src)

        loss = binary_auprc(
            pred.flatten(),
            label.flatten(),
        )
        losses += loss.item()

    return losses / len(val_dataloader)


def train_epoch(model, optimizer, prot_data, loss_fn):
    model.train()
    losses = 0
    train_dataloader = DataLoader(
        prot_data.train_iter,
        batch_size=BATCH_SIZE,
        collate_fn=prot_data.collate_fn
    )

    for src, label in train_dataloader:
        src = src.to(DEVICE)
        label = label.to(DEVICE)

        pred = model(src)

        optimizer.zero_grad()
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def train_model(model, prot_data, model_name):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_aupr = 0
    weights_path = f'weights/best_weights_{model_name}.pt'

    for epoch in range(1, NUM_EPOCHS+1):
        print(f"Training Epoch {epoch}")
        start_time = timer()
        train_loss = train_epoch(model, optimizer, prot_data, loss_fn)
        end_time = timer()
        val_loss = evaluate(model, loss_fn, prot_data)
        val_aupr = evaluate_aupr(model, prot_data)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Val AUPR: {val_aupr:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if val_aupr > best_aupr:
            print('Validation AUPR is higher than best AUPR, saving...')
            best_aupr = val_aupr
            torch.save(model.state_dict(), weights_path)
        else:
            print('Validation AUPR not higher than best AUPR')
