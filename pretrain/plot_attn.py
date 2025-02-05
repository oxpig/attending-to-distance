import math
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import optimize
from matplotlib import pyplot as plt

from models.plottable_model import BERTCoords
from masked_prot_data import MaskedProtData

from const import BATCH_SIZE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d_model = 768
num_heads = 12
num_layers=6


def gaussian(x, mean, amplitude, stddev):
    return amplitude * np.exp(-((np.array(x-mean)) / (math.sqrt(2) * stddev))**2)


def gaussian0(x, amplitude, stddev):
    return amplitude * np.exp(-((np.array(x)) / (math.sqrt(2) * stddev))**2)


# Inefficient implentation
def lin_dist(coords):
    seq_len = coords.shape[0]
    dist_mat = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            # dist_mat[i,j] = abs(i-j)
            dist_mat[i,j] = i-j
    return dist_mat


def plot_lin_3d(results_path, lin_x=True):
    val_data = MaskedProtData('test', shuffle=False, mask=False)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    idx = 0
    max_idx = 1 # Arbitrary cutoff (can be small)
    all_lin_dists = []
    all_dists_3d = []
 
    for data in val_dataloader:
        idx += 1
        print(f'Running batch {idx}/{max_idx}')
        src_pos = data['coords'].to(DEVICE).transpose(0,1)
        prots = data['prot']
        coords = src_pos[:,:,:3].detach().cpu().numpy()
        l, bs, _ = coords.shape
        mask = data['mask'].to(DEVICE)
 
        for i in range(bs):
            prot_coords = coords[:,i,:]
            prot_mask = np.any(prot_coords != 0, 1) # All zeros means no pos
            lin_dists = lin_dist(prot_coords)
            dists_3d = squareform(pdist(prot_coords))
            lin_dists = lin_dists[:,prot_mask]
            lin_dists = lin_dists[prot_mask,:]
            dists_3d = dists_3d[:,prot_mask]
            dists_3d = dists_3d[prot_mask,:]
            dist_mask = lin_dists < 21 if lin_x else dists_3d < 21
            lin_dists = lin_dists[dist_mask]
            dists_3d = dists_3d[dist_mask]
            if not lin_x:
                dists_3d = dists_3d.round().astype(int)
            lin_dists = lin_dists.astype(int)
            lin_dists = list(lin_dists.flatten())
            dists_3d = list(dists_3d.flatten())
            all_lin_dists += lin_dists
            all_dists_3d += dists_3d
 
        if idx == max_idx:
            break

    amplitudes, std_devs = [], []
    plt.clf()
    new_data = [[] for _ in range(22)]
    for i in range(len(all_dists_3d)):
        this_lin_dist = all_lin_dists[i]
        this_dist_3d = all_dists_3d[i]
        if lin_x:
            # new_data[this_lin_dist].append(this_dist_3d)
            new_data[this_lin_dist].append(math.exp(-this_dist_3d**2/100))
        else:
            new_data[this_dist_3d].append(math.exp(-this_lin_dist**2/10))
    for i in range(len(new_data)):
        if len(new_data[i]) < 100:
            new_data[i] = []

    plt.boxplot(
        new_data,
        showfliers=False,
        labels=list(range(len(new_data)))
    )
    if lin_x:
        plt.xlabel('Sequence distance')
        plt.ylabel('3D distance (Å)')
    else:
        plt.ylabel('Sequence distance')
        plt.xlabel('3D distance (Å)')
    plt.savefig(results_path)


def plot_residue_attns(model, val_data, results_path):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    idx = 0
    max_idx = 1 # Arbitrary cutoff (can be small)
    tmap = val_data.tok2char
    res_attns = [[[[] for _ in tmap] for _ in tmap] for _ in range(num_layers)]
 
    for data in val_dataloader:
        idx += 1
        print(f'Running batch {idx}/{max_idx}')
        src = data['in_seq'].to(DEVICE).transpose(0,1)
        src_pos = data['coords'].to(DEVICE).transpose(0,1)
        prots = data['prot']
        coords = src_pos[:,:,:3].detach().cpu().numpy()
        l, bs, _ = coords.shape
        mask = data['mask'].to(DEVICE)
 
        logits, attn_weights = model(src, src_pos, mask)
        for i in range(bs):
            prot_coords = coords[:,i,:]
            prot_mask = np.any(prot_coords != 0, 1) # All zeros means no pos
            src_toks = src[:,i][prot_mask]
            for j in range(num_layers):
                layer_weights = attn_weights[j].detach().cpu().numpy()
                head_weights = layer_weights[i,...]
                head_weights = head_weights[:,prot_mask,:]
                head_weights = head_weights[:,:,prot_mask]
                head_weights = head_weights.mean(0)
                for aa1_idx in range(len(src_toks)):
                    aa1 = src_toks[aa1_idx]
                    aa1_attns = [[] for _ in tmap]
                    for aa2_idx in range(len(src_toks)):
                        aa2 = src_toks[aa2_idx]
                        aa1_attns[aa2].append(head_weights[aa1_idx, aa2_idx])
                    for aa2 in range(len(aa1_attns)):
                        res_attns[j][aa1][aa2].append(sum(aa1_attns[aa2])/(len(aa1_attns[aa2]) or 1))
 
        if idx == max_idx:
            break

    for i in range(num_layers):
        plt.clf()
        attns = [[sum(res_attns[i][aa1][aa2])/(len(res_attns[i][aa1][aa2]) or 1) for aa1 in tmap] for aa2 in tmap]
        plt.imshow(attns)
        plt.xlabel('Query residue')
        plt.ylabel('Key residue')
        plt.xticks(np.arange(len(tmap)), labels=[tmap[i] for i in range(len(tmap))])
        plt.yticks(np.arange(len(tmap)), labels=[tmap[i] for i in range(len(tmap))])
        plt.colorbar()
        plt.title(f'Layer {i+1}')
        plt.savefig(f'{results_path}_layer_{i:02}_attentions.png')


def plot_attentions(model, val_data, results_path, lin=False):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)
    dist_attns = {i: defaultdict(list) for i in range(num_layers)}
    idx = 0
    max_idx = 1 # Arbitrary cutoff (can be small)
 
    for data in val_dataloader:
        idx += 1
        print(f'Running batch {idx}/{max_idx}')
        src = data['in_seq'].to(DEVICE).transpose(0,1)
        src_pos = data['coords'].to(DEVICE).transpose(0,1)
        prots = data['prot']
        coords = src_pos[:,:,:3].detach().cpu().numpy()
        l, bs, _ = coords.shape
        mask = data['mask'].to(DEVICE)
 
        logits, attn_weights = model(src, src_pos, mask)
        for i in range(bs):
            prot_coords = coords[:,i,:]
            prot_mask = np.any(prot_coords != 0, 1) # All zeros means no pos
            if lin:
                dists = lin_dist(prot_coords)
            else:
                dists = squareform(pdist(prot_coords))
            dists = dists[:,prot_mask]
            dists = dists[prot_mask,:]
            dists = dists.round().astype(int)
            dist_mask = (-11 < dists) & (dists < 11) if lin else dists < 21
            dists = dists[dist_mask]
            for j in range(num_layers):
                layer_weights = attn_weights[j].detach().cpu().numpy()
                head_weights = layer_weights[i,...]
                head_weights = head_weights[:,prot_mask,:]
                head_weights = head_weights[:,:,prot_mask]
                head_weights = head_weights[2]
                head_weights = head_weights[dist_mask]
                dists = dists.flatten()
                head_weights = head_weights.flatten()
                for k in range(len(dists)):
                    dist = dists[k]
                    attn = head_weights[k] 
                    dist_attns[j][dist].append(attn)
 
        if idx == max_idx:
            break

    amplitudes, std_devs = [], []
    for i in range(num_layers):
        plt.clf()
        dists = sorted(list(dist_attns[i].keys()))
        # Unusual distances correspond to erroneous structures
        dists = [dist for dist in dists if len(dist_attns[i][dist]) > 100]
        x = list(range(-10, 11)) if lin else list(range(21))
        attns = [sum(dist_attns[i][dist])/len(dist_attns[i][dist]) for dist in dists]
        all_attns = [dist_attns[i][dist] for dist in dists]
        plt.boxplot(all_attns, showfliers=False, labels=dists, positions=dists)
        fit_fn = gaussian if lin else gaussian0
        try:
            popt, _ = optimize.curve_fit(fit_fn, dists, attns)
            if lin:
                mean, amplitude, std_dev = popt
            else:
                amplitude, std_dev = popt
            amplitudes.append(amplitude)
            std_devs.append(std_dev)
            plt.plot(x, fit_fn(x, *popt), label='Gaussian fit')
        except RuntimeError:
            print('Couldn\'t fit Gaussian')
            amplitudes.append(None)
            std_devs.append(None)
        if lin:
            plt.xlabel('Distance (residues)', fontsize=30)
        else:
            plt.xlabel('Distance (Å)', fontsize=30)
        # plt.ylim(0,0.2)
        plt.ylim(0,0.5)
        # plt.ylim(0,0.4)
        plt.ylabel('Average attention', fontsize=30)
        plt.title(f'Layer {i+1}', fontsize=40)
        plt.tight_layout()
        plt.savefig(f'{results_path}_layer_{i:02}_attentions.png')
    return amplitudes, std_devs


def plot_model_attentions(model_path, add_coords=True, lin=False, raw=True):
    test_data = MaskedProtData('valid', shuffle=False, mask=False, all_a=not raw)
    vocab_size = len(test_data.char2tok.keys())
    model = BERTCoords(
        num_encoder_layers=num_layers,
        emb_size=d_model,
        nhead=num_heads,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dropout=0,
        add_coords=add_coords,
        remove_lin=not lin and not raw,
        remove_3d=lin and not raw,
    )
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_type = 'coords' if add_coords else 'no_coords'
    dist = 'linear' if lin else '3d'
    if raw:
        results_path = f'results/attention_dist_plots/raw/{model_type}_{dist}'
    else:
        results_path = f'results/attention_dist_plots/{model_type}_{dist}'
    return plot_attentions(model, test_data, results_path, lin=lin)


def plot_model_res_attentions(model_path, add_coords=True):
    test_data = MaskedProtData('valid', shuffle=False, mask=False)
    vocab_size = len(test_data.char2tok.keys())
    model = BERTCoords(
        num_encoder_layers=num_layers,
        emb_size=d_model,
        nhead=num_heads,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dropout=0,
        add_coords=add_coords,
        remove_lin=True,
        remove_3d=True,
    )
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_type = 'coords' if add_coords else 'no_coords'
    results_path = f'results/attention_dist_plots/raw/{model_type}_res'
    return plot_residue_attns(model, test_data, results_path)


def plot_comparison():
    a1, d1 = plot_model_attentions("weights/best_weights_bert_coords.pt", True)
    a2, d2 = plot_model_attentions("weights/best_weights_bert_no_coords.pt", False)
    a3, d3 = plot_model_attentions("weights/best_weights_bert_coords.pt", True, True)
    a4, d4 = plot_model_attentions("weights/best_weights_bert_no_coords.pt", False, True)
    plot_lin_3d('results/lin_3d_box.pdf')
    plot_lin_3d('results/3d_lin_box.pdf', lin_x=False)
    layers = [i+1 for i in range(num_layers)]

    plt.clf()
    plt.plot(layers, a1, label='With coords, 3D')
    plt.plot(layers, a3, label='With coords, linear')
    plt.plot(layers, a4, label='No coords, linear')
    plt.xlabel('Layer index', fontsize=16)
    plt.ylabel('Attention amplitude', fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('results/layer_amplitudes.pdf')

    plt.clf()
    plt.plot(layers, d1, label='With coords, 3D')
    plt.plot(layers, d3, label='With coords, linear')
    plt.plot(layers, d4, label='No coords, linear')
    plt.xlabel('Layer index', fontsize=16)
    plt.ylabel('Attention std deviation', fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('results/layer_std_devs.pdf')

plot_comparison()
# plot_model_res_attentions("weights/best_weights_bert_coords.pt", True)
# plot_model_res_attentions("weights/best_weights_bert_coords.pt", False)
