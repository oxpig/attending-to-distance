import math
from timeit import default_timer as timer

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

from const import BATCH_SIZE, NUM_EPOCHS
from noam_opt import get_std_opt
from metric import distance


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_label(pos):
    n_nodes = 5
    label = np.array([
        [distance(pos[i], pos[j], 2) for i in range(n_nodes)]
        for j in range(n_nodes)]
    )
    return label


def recentre(src_pos):
    # Recentre all points about centre of mass
    src_pos = src_pos - src_pos.mean(0, keepdim=True)
    return src_pos


def random_transform(src_pos):
    src_pos = recentre(src_pos)
    n_pts, batch_sz, dim = src_pos.shape
    new_pos = torch.empty(src_pos.shape, dtype=src_pos.dtype)
    for j in range(src_pos.shape[1]):
        # Some rotation
        r = R.random()
        for i in range(src_pos.shape[0]):
            new_vec = r.apply(src_pos[i,j].cpu().numpy())
            for k in range(src_pos.shape[2]):
                new_pos[i,j,k] = new_vec[k]
    rand_shift = 0*torch.randn((1, batch_sz, dim))
    return new_pos + rand_shift


def evaluate(model, loss_fn, pt_data, test=False, print_first=False):
    model.eval()
    losses = 0
 
    val_dataloader = DataLoader(
        pt_data.val_iter,
        batch_size=BATCH_SIZE,
        collate_fn=pt_data.collate_fn
    )

    is_first = True
    for src, src_pos, label in val_dataloader:
        src_pos = recentre(src_pos)
        src = src.to(DEVICE)
        src_pos = src_pos.to(DEVICE)
        label = label.to(DEVICE)
 
        pred = model(src, src_pos, test)
        if print_first and is_first:
            print(pred[0])
            print(label[0])
            is_first = False
 
        loss = loss_fn(pred, label)
        losses += loss.item()
 
    return losses / len(val_dataloader)


def train_epoch(model, optimizer, pt_data, loss_fn, augment=False):
    model.train()
    losses = 0
    train_dataloader = DataLoader(
        pt_data.train_iter,
        batch_size=BATCH_SIZE,
        collate_fn=pt_data.collate_fn
    )

    for src, src_pos, label in train_dataloader:
        src_pos = recentre(src_pos)
        if augment:
            if src_pos.shape[-1] == 6:
                src_pos = torch.cat((
                    random_transform(src_pos[:,:,:3]),
                    random_transform(src_pos[:,:,3:]),
                ), -1)
            else:
                src_pos = random_transform(src_pos)
        src = src.to(DEVICE)
        src_pos = src_pos.to(DEVICE)
        label = label.to(DEVICE)

        pred = model(src, src_pos)

        optimizer.zero_grad()

        loss = loss_fn(pred, label)
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def plot_loss(epochs, train_losses, val_losses, plot_path):
    from matplotlib import pyplot as plt
    import pandas as pd
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    df.to_csv(plot_path.replace('.pdf', '.csv').replace('/plots/', '/data/'), index=False)
    plt.clf()
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(plot_path)


def train_model(model, pt_data, save_best=True, plots_path=None, augment=False):
    loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    optimizer = get_std_opt(model.parameters(), model.d_model)

    best_loss = math.inf
    val_losses = []
    train_losses = []
    epochs = list(range(1, NUM_EPOCHS+1))

    for epoch in epochs:
        print(f"Training Epoch {epoch}")
        start_time = timer()
        train_loss = train_epoch(model, optimizer, pt_data, loss_fn, augment)
        end_time = timer()
        val_loss = evaluate(model, loss_fn, pt_data)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if val_loss < best_loss:
            best_loss = val_loss
            if save_best:
                print('Validation loss is lower than best loss, saving...')
                torch.save(model.state_dict(), 'models/best_weights.pt')
    if plots_path is not None:
        plot_loss(epochs, train_losses, val_losses, plots_path)

    return best_loss
