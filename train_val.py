import torch
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm

from my_utils import get_au_fa_fr, count_FA_FR


def train_epoch(model, opt, loader, melspec, gru_nl, hidden_size, device):
    model.train()
    for i, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = torch.log(melspec(batch) + 1e-9).to(device)

        opt.zero_grad()

        # define frist hidden with 0
        hidden = torch.zeros(gru_nl*2, batch.size(0), hidden_size).to(device)    # (num_layers*num_dirs,  BS, HS)
        # run model
        probs = model(batch, hidden)
        loss = F.nll_loss(probs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        opt.step()

        # logging
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.true_divide(
                            torch.sum(argmax_probs == labels),
                            torch.numel(argmax_probs)
        )
        wandb.log({'loss':loss.item(), 'train_FA':FA, 'train_FR':FR, 'train_acc':acc})


def validation(model, loader, melspec, gru_nl, hidden_size, device, find_trsh=False):
    model.eval()
    with torch.no_grad():
        val_losses, accs, FAs, FRs = [], [], [], []
        all_probs, all_labels = [], []
        for i, (batch, labels) in tqdm(enumerate(loader)):
            batch, labels = batch.to(device), labels.to(device)
            batch = torch.log(melspec(batch) + 1e-9).to(device)

            # define frist hidden with 0
            hidden = torch.zeros(gru_nl*2, batch.size(0), hidden_size).to(device)    # (num_layers*num_dirs,  BS, HS)
            # run model
            probs = model(batch, hidden)
            loss = F.nll_loss(probs, labels)

            # logging
            argmax_probs = torch.argmax(probs, dim=-1)
            all_probs.append(torch.exp(probs)[:, 1])
            all_labels.append(labels)
            val_losses.append(loss.item())
            accs.append(torch.true_divide(
                                torch.sum(argmax_probs == labels),
                                torch.numel(argmax_probs)).item()
                       )
            FA, FR = count_FA_FR(argmax_probs, labels)
            FAs.append(FA)
            FRs.append(FR)

        # area under FA/FR curve for whole loader
        best_trsh, au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0), all_labels, device, find_trsh=True)
        wandb.log({'mean_val_loss':np.mean(val_losses), 'mean_val_acc':np.mean(accs),
                   'mean_val_FA':np.mean(FAs), 'mean_val_FR':np.mean(FRs),
                   'au_fa_fr':au_fa_fr})
