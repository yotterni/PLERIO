import os
import sys

import pandas as pd

import torch
import esm

import numpy as np
import random
import jax_unirep as urp

import pickle


def esm_embedding(model_type: esm.pretrained, epoch_num: int,
                  device: str, data: list[tuple[str, str]]) -> list[torch.Tensor]:
    model, alphabet = model_type()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[epoch_num],
                        return_contacts=False)
    token_representations = results["representations"][epoch_num]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1 : tokens_len - 1].mean(0))
    return sequence_representations


def calc_unirep_embeddings() -> list[torch.Tensor]:
    mapped_prots = pd.read_csv('../prot_embeddings/mapped_prots.tsv',
                               sep='\t')
    name_to_emb = {}
    data = list(zip(mapped_prots['From'], mapped_prots['Sequence']))

    for name, seq in data:
        name_to_emb[name] = torch.Tensor(urp.get_reps(seq)[1][0]).view(1, -1)

    with open('../prot_embeddings/unirep_embs.pkl', 'wb') as file:
        pickle.dump(name_to_emb, file=file, protocol=5)


def calc_prot_embeddings() -> None:
    mapped_prots = pd.read_csv('../prot_embeddings/mapped_prots.tsv',
                               sep='\t')
    mapped_prots.drop_duplicates(subset='From', keep='first', inplace=True)

    device = 'cpu'

    model = esm.pretrained.esm2_t33_650M_UR50D
    name_to_emb = {}
    data = list(zip(mapped_prots['From'], mapped_prots['Sequence']))
    for item in data:
        emb = esm_embedding(model, 33, device, [item])
        name_to_emb[item[0]] = emb[0].view(1, -1)

    with open('../prot_embeddings/prot_embs.pkl', 'wb') as file:
        pickle.dump(name_to_emb, file)


def calculate_protein_weights(cell_line: str) -> None:
    pref = f'single_prot_dbs/{cell_line}'
    dataframes = [(protein,
                   pd.read_csv(f'{pref}/{protein}/{protein}.narrowPeak'))
                  for protein in os.listdir(pref)]
    peak_numbers = list(map(lambda x: (x[0], len(x[1])), dataframes))
    peak_numbers = sorted(peak_numbers, key=lambda x: x[1])
    peaks_total = sum(x[1] for x in peak_numbers)

    protein_freqs = [(x[0], x[1] / peaks_total) for x in peak_numbers]
    max_frequency = max(protein_freqs, key=lambda x: x[1])[1]
    protein_weights = [(x[0], max_frequency / x[1]) for x in protein_freqs]
    with open(f'multi_prot_dbs/{cell_line}/protein_frequency_weights.txt',
              'w') as file:
        for pair in protein_weights:
            print(*pair, file=file)

if __name__ == '__main__':
    if sys.argv[1] == 'esm':
        calc_prot_embeddings()
    else:
        calc_unirep_embeddings()
