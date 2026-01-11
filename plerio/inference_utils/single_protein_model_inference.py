import sys
sys.path.append('')

from plerio.engine.dataconstruction_utils.kmer_counting import KmerCounter
from matplotlib.colors import ListedColormap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn

import typing as tp

device = 'cpu'

def cpu_num(tns):
    return tns.cpu().detach().numpy()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))

def interpolate_color(color1, color2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(color1, color2))

def generate_gradient(hex1, hex2, steps=256):
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    return [interpolate_color(rgb1, rgb2, t / (steps - 1)) for t in range(steps)]

def create_colormap(hex_start, hex_end, steps=256):
    gradient_rgb = generate_gradient(hex_start, hex_end, steps)
    gradient_rgb_normalized = [(r/255, g/255, b/255) for r, g, b in gradient_rgb]
    return ListedColormap(gradient_rgb_normalized)

class PretrainedProteinModel:
    def __init__(self,
                 protein_name: str,
                 cell_line: str,
                 weight_path: str | Path,
                 step: tp.Optional[int] = 50,
                 device: str | torch.device = 'cpu') -> None:
        """
        Initializes the pretrained model class.
        :param protein_name: the name of the protein. Note that this
         parameter is necessary only for metrics plotting.
        :param cell_line: the name of the cell line. Note that this
         parameter is necessary only for metrics plotting.
        :param weight_path: the path of the weights of pretrained model.
        :param step: step of the sliding window. Model can process only
         a window 200 nucleotides long, so for longer sequences step
         is required to determine which stride should be used to move
         the window to the next point.
        :param device: the device for model running. The lack of speed
         on the CPU in comparison to GPU can be encountered only on the
         long seuquences.
        """
        self.window_size = 200
        self.protein_name = protein_name
        self.cell_line = cell_line
        self.step = step
        self.device= 'cpu'
        self.model = nn.Sequential(
            nn.Linear(1088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(weight_path, weights_only=True, map_location=device))

        self.model.eval()
        self.kmer_counter = KmerCounter(ks=[3, 5],
                                        torch_output=True,
                                        rna_alphabet=True)
        self.cmap = create_colormap('#FFFFFF', '#237AE0')

    def __call__(self, seq: str, plot_result: bool = True) -> np.ndarray:
        bin_predictions = []
        end = len(seq) - self.window_size + 1
        if len(seq) < self.window_size:
            end = len(seq)
        for start in range(0, end, self.step):
            current_seq = seq[start:start + self.window_size]
            emb = self.kmer_counter(current_seq)
            emb = emb.to(device)
            bin_pred = self.model(emb)
            bin_predictions.append(cpu_num(bin_pred))

        track = np.array(bin_predictions)
        if plot_result:
            self.plot_track(track)
        return track.reshape(track.shape[0])

    def plot_track(self, track: np.ndarray) -> None:
        plt.figure(0, (10, 1))
        ax = sns.heatmap(track.T, vmin=0, vmax=1, cmap=self.cmap)


        if len(track) < 10:
            num_ticks = len(track)
        else:
            num_ticks = 10

        xticks = np.linspace(0, len(track), num_ticks)
        xtick_labels = xticks * self.step
        xtick_labels = xtick_labels.astype(int)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        plt.setp(ax.get_xticklabels(), rotation=30)

        ax.set(xlabel='BPs from the start of the RNA')

        # plt.legend(title='Probability', loc='upper right')
        plt.title(f'Probabilities of {self.protein_name} protein binding '
                  f'along the given RNA in {self.cell_line}')
