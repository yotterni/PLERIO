import os
import pickle
import typing as tp
import warnings

import numpy as np
import random

import torch
import torch.nn as nn

from sklearn.metrics import (balanced_accuracy_score,
                             roc_auc_score,
                             matthews_corrcoef)

# from custom_protein_weighted_loss import OneSideWeightedBCEWithLogitsLoss


# TODO: move it to the config
warnings.filterwarnings("ignore")
random.seed(42)
torch.manual_seed(42)
DROP_PROB = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpu_num(array):
    return array.cpu().detach().numpy()


def beautiful_printout(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'{args[1]} loss: {result[0]:.04f}')
        print(f'{args[1]} bal_acc: {result[1]:.04f}')
        print(f'{args[1]} roc_auc: {result[2]:.04f}')
        print(f'{args[1]} mcc: {result[3]:.04f}')
        return result
    return inner


@beautiful_printout
def run_epoch(cell_line: str, fold: str, model: nn.Sequential,
              prot_embeddings: dict[str, torch.Tensor],
              loss_fn: tp.Any, optimizer: tp.Any,
              prot_to_weight: dict[str, float],
              history: dict[str, list[float]],
          ) -> tuple[float, float, float, float]:
    """
    A function that applies a single learning epoch to the model.
    It expects that locations of the train and val dataset are as follows:
    f'multi_prot_dbs/{cell_line}/train_dataset/'
    :param cell_line: cell line name, e.g. K562 or HepG2
    :param fold: 'train' or 'val'
    :param model: model that is capable of consuming data from datasets
    :param loss_fn: loss function for binary classification
    :param optimizer: an optimizer to optimize the model
    :param prot_to_weight: mapping from protein names to protein weights
    calculated based on their frequencies
    :param prot_embeddings: mapping from protein names to protein embeddings
    :param history: mapping from metric name to it's history
    :return: tuple of (loss, bal_acc, roc_auc, mcc)
    """
    assert fold in ['train', 'val'], 'Fold must be either train or val!'

    dataset_path = f'multi_prot_dbs/{cell_line}/{fold}_dataset/'
    files = os.listdir(dataset_path)
    random.shuffle(files)

    losses, bal_accs, roc_aucs, mccs = [], [], [], []

    for filename in files:
        with open(dataset_path + filename, 'rb') as file:
            rna_batch, prot_names, y_true = pickle.load(file)

        weight_batch = torch.Tensor([prot_to_weight[prot_name]
                                     for prot_name in prot_names])
        weight_batch /= sum(weight_batch)
        weight_batch = weight_batch.view(-1, 1)

        prot_batch = torch.cat([prot_embeddings[prot_name]
                          for prot_name in prot_names], dim=0)

        pair_batch = torch.cat([rna_batch, prot_batch], dim=1)

        pair_batch= pair_batch.to(device)
        weight_batch = weight_batch.to(device)
        y_true = y_true.to(device)

        if fold == 'train':
            model.train()
            optimizer.zero_grad()
            y_pred = model(pair_batch)
            loss = loss_fn(y_pred, y_true, weight_batch)
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                y_pred = model(pair_batch)
                loss = loss_fn(y_pred, y_true, weight_batch)

        y_pred = y_pred.sigmoid()
        # TODO: handle NaN more bal_accurate
        y_pred = y_pred.nan_to_num()
        probabilities = cpu_num(y_pred)
        y_true = cpu_num(y_true)

        losses.append(loss.item())
        bal_accs.append(balanced_accuracy_score(y_true, probabilities >= 0.5))
        roc_aucs.append(roc_auc_score(y_true, probabilities))
        mccs.append(matthews_corrcoef(y_true, probabilities >= 0.5))

    history['loss'].append(np.mean(losses))
    history['bal_acc'].append(np.mean(bal_accs))
    history['roc_auc'].append(np.mean(roc_aucs))
    history['mcc'].append(np.mean(mccs))
    return (history['loss'][-1], history['bal_acc'][-1],
            history['roc_auc'][-1], history['mcc'][-1])


def parse_protein_frequencies(file: str) -> dict[str, float]:
    """
    Parse a file with protein names and their frequency weights
    to the dictionary from protein to corresponding weight.
    :param file: path-like, absolute or relative path to a file
    to parse. If the path is relative,
    :return: mapping from protein names to their weights
    """
    prot_to_weight = {}
    with open(file, 'r') as file:
        for line in file:
            prot, weight = line.split()
            prot_to_weight[prot] = float(weight)
    return prot_to_weight


def train_sequential_model(cell_line: str,
                           epochs: int = 10,
                           positive_class_weight: float = 0.5,
                           learning_rate: float = 5e-3,
                           weight_decay: float = 1e-7,
                           scheduler_gamma: float = 0.8) -> None:
    """
    Initializes and trains a simple sequential model.
    :param cell_line: cell line name, e.g. K562 or HepG2
    :param epochs: number of training epochs (full iterations over
    train and val datasets)
    :param positive_class_weight: weight for positive class
    :param learning_rate: learning rate which will be passed to optimizer
    :param weight_decay: regularization coefficient for optimizer
    :param scheduler_gamma: gamma for exponential scheduler
    :return: None
    """
    # TODO: strict definition for positive class weight
    with open('prot_embeddings/unirep_embs.pkl', 'rb') as file:
        prot_embeddings = pickle.load(file)
        prot_emb_size = prot_embeddings[list(prot_embeddings.keys())[0]].shape[1]

    prot_to_weight = parse_protein_frequencies(
        f'multi_prot_dbs/{cell_line}/protein_frequency_weights.txt')

    model = nn.Sequential(nn.Linear(1088 + prot_emb_size, 512),
                          nn.ReLU(),
                          nn.Dropout(DROP_PROB),
                          nn.Linear(512, 256),
                          nn.ReLU(),
                          nn.Dropout(DROP_PROB),
                          nn.Linear(256, 128),
                          nn.ReLU(),
                          nn.Dropout(DROP_PROB),
                          nn.Linear(128, 1)).to(device)

    loss_fn = OneSideWeightedBCEWithLogitsLoss(positive_class_weight)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=scheduler_gamma)

    train_history = {'loss': [], 'bal_acc': [], 'roc_auc': [], 'mcc': []}
    val_history = {'loss': [], 'bal_acc': [], 'roc_auc': [], 'mcc': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}.................:)')
        run_epoch(cell_line, 'val', model, prot_embeddings,
                  loss_fn, optimizer, prot_to_weight, val_history)
        run_epoch(cell_line, 'train', model, prot_embeddings,
                  loss_fn, optimizer, prot_to_weight, train_history)
        scheduler.step()

        torch.save(model.state_dict(),
                   f'models/multi_prot_models/{cell_line}/FCNN_model.pt')


if __name__ == '__main__':
    train_sequential_model('K562', 7, 2, 1e-3)
    train_sequential_model('HepG2', 7, 2, 1e-3)
