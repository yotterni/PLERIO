# PLERIO: Protein - [low expressed] RNA interaction oracle

## Introduction
The study of protein-RNA interactions (RPIs) is critical for understanding many cellular processes. Existing experimental protocols for RPI discovery are biased toward highly expressed RNAs because no normalization is applied to the expression rates of RNAs. However, many regulatory RNAs are low-expressed and information about them is likely to be missing from the experimental data. It is therefore hoped that deep learning models will be able to find binding patterns between certain proteins and RNAs, and can be used to screen low-expressed RNAs for potential protein-binding ones.

## TL;DR: simple usage example in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lgR05mLUk1lHpKSxCB76bvcMGGj6mHzi?usp=sharing)

## Graphical representation of the work

![assets/plerio_poster](https://github.com/user-attachments/assets/cd0111d9-c9a3-49ef-b8df-144eeae5eeae)

## Installation
```bash
git clone https://github.com/Yotterni/PLERIO.git
pip install -e PLERIO
```

## Usage
First, you need to import pretrained model class from `plerio` library.

```python
from plerio.engine.inference_utils.single_protein_model_inference import PretrainedProteinModel
```
Then you should select the desired protein and cell line for which you want to import the model. You will also need to manually specify the location of the model weights. Model weights were downloaded during the installation process, so now you just need to understand where the `single_prot_models` folder is located according to your current directory. Then, when all parameters are determined, initialize the instance of the pretrained model class.

```python

PROTEIN_NAME = 'FASTKD2'
CELL_LINE = 'K562'
PATH_TO_WEIGHTS = 'single_prot_models/K562/FASTKD2/FCNN_model.pt'
MY_RNA_SEQUENCE = ...

model = PretrainedProteinModel(
    protein_name=PROTEIN_NAME,
    cell_line=CELL_LINE,
    weight_path=PATH_TO_WEIGHTS
)

```
Now you are invited to pass your RNA sequence to our model and get predictions for each of the 200 nucleotide bins of this RNA. If the RNA is shorter than 200 nt, the output will contain only one bin. Note that we use the `AUGC` alphabet, i.e. sequences containing T instead of U will be processed incorrectly. Also note that the model expects the sequence to be expressed from the `+` strand of the DNA, so you may want to process your RNA with `reverse_complement` from the `Bio` package or similar.

```python
model(rna_sequence) # plot_result=False if you dont want to plot sequence heatmap.
```
