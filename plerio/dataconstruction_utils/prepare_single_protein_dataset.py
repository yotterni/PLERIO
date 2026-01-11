# import numpy as np
# import pandas as pd

# import pickle

# import torch

# import subprocess

# from Bio import SeqIO




# def prepare_single_protein_dataset(protein_name: str,
#                                    cell_line: str,
#                                    window_size: int,
#                                    batch_size: int = 8192) -> None:
#     print(f'Processing {protein_name}...')
#     positive_df = pd.read_csv(
#         f'single_prot_dbs/{cell_line}/{protein_name}/{protein_name}.narrowPeak',
#                               names=['chr', 'start', 'end', 'protname',
#                                      'darkness', 'strand', 'signal',
#                                      'pval', 'wtf1', 'wtf2'],
#                               sep='\t')

#     enlarged_positive_df = augment_peaks(enlarge_peaks(positive_df, window_size))
#     enlarged_positive_df.to_csv(
#         f'single_prot_dbs/{cell_line}/{protein_name}/positives_enlarged.tsv',
#                                 sep='\t', header=False, index=False)

#     subprocess.call(f'bedtools subtract -a genes_set5.bed '
#                     f'-b single_prot_dbs/{cell_line}/{protein_name}/positives_enlarged.tsv '
#                     f'> single_prot_dbs/{cell_line}/{protein_name}/negatives.bed', shell=True)

#     subprocess.call(
#         f'bedtools intersect '
#         f'-a single_prot_dbs/{cell_line}/{protein_name}/positives_enlarged.tsv'
#         f' -b single_prot_dbs/{cell_line}/{protein_name}/negatives.bed'
#         f'> single_prot_dbs/{cell_line}/{protein_name}/no_intersection.bed', shell=True)

#     negative_df = pd.read_csv(
#         f'single_prot_dbs/{cell_line}/{protein_name}/negatives.bed', sep='\t',
#                               names=['chr', 'start', 'end',
#                                      'strand', 'annotation', 'gene_name',
#                                      'altern_name', 'source'])

#     negative_peaks_df = generate_negatives(negative_df, window_size)

#     enlarged_positive_df['label'] = [1] * len(enlarged_positive_df)
#     positive_peaks_df = pd.concat(
#         [enlarged_positive_df[['chr', 'start', 'end',
#                                'strand', 'label']]],
#         ignore_index=True)

#     final_df = pd.concat([positive_peaks_df,
#                           negative_peaks_df.sample(n=len(positive_peaks_df))], axis=0)
#     final_df = final_df.sample(frac=1)

#     print(f"Protein: {protein_name}, positive fraction: {final_df['label'].mean()}")

#     rna_train_set = {'chr1', 'chr3', 'chr4', 'chr6',
#                      'chr7', 'chr9', 'chr11', 'chr12',
#                      'chr13', 'chr14', 'chr15', 'chr16',
#                      'chr18', 'chr20', 'chr21', 'chrX'}

#     rna_val_set = {'chr5', 'chr8', 'chr17', 'chr22'}
#     rna_test_set = {'chr2', 'chr10', 'chrY', 'chr19'}

#     train_coord = final_df[final_df['chr'].isin(rna_train_set)]
#     val_coord = final_df[final_df['chr'].isin(rna_val_set)]
#     test_coord = final_df[final_df['chr'].isin(rna_test_set)]

#     train_coord.to_csv(f'single_prot_dbs/{cell_line}/{protein_name}/train_coord.bed',
#                        sep='\t', index=False, header=False)
#     val_coord.to_csv(f'single_prot_dbs/{cell_line}/{protein_name}/val_coord.bed',
#                      sep='\t', index=False, header=False)
#     test_coord.to_csv(f'single_prot_dbs/{cell_line}/{protein_name}/test_coord.bed',
#                       sep='\t', index=False, header=False)

#     genome_fasta = SeqIO.to_dict(SeqIO.parse("hg38.fna", "fasta"))
#     cnt = 0
#     chrnames = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM']

#     ##
#     ## РАСКОММЕНТИТЬ, ЕСЛИ НА ДОМАШНЕМ ХГ38

#     for key in list(genome_fasta.keys()):
#         seq = genome_fasta.pop(key)
#         if 'NC' in key:
#             # print(key)
#             genome_fasta[chrnames[cnt]] = seq
#             cnt += 1


#     ##
#     ##

#     seq_extractor(
#         f'single_prot_dbs/{cell_line}/{protein_name}/train_coord.bed',
#         genome_fasta)
#     seq_extractor(
#         f'single_prot_dbs/{cell_line}/{protein_name}/val_coord.bed',
#         genome_fasta)
#     seq_extractor(
#         f'single_prot_dbs/{cell_line}/{protein_name}/test_coord.bed',
#         genome_fasta)

#     try:
#         subprocess.call(
#             f'rm -r single_prot_dbs/{cell_line}/{protein_name}/train_dataset',
#             shell=True)
#         subprocess.call(
#             f'rm -r single_prot_dbs/{cell_line}/{protein_name}/val_dataset',
#             shell=True)
#         subprocess.call(
#             f'rm -r single_prot_dbs/{cell_line}/{protein_name}/test_dataset',
#             shell=True)
#     except subprocess.CalledProcessError:
#         pass

#     subprocess.call(
#         f'mkdir single_prot_dbs/{cell_line}/{protein_name}/train_dataset',
#         shell=True)
#     subprocess.call(
#         f'mkdir single_prot_dbs/{cell_line}/{protein_name}/val_dataset',
#         shell=True)
#     subprocess.call(
#         f'mkdir single_prot_dbs/{cell_line}/{protein_name}/test_dataset',
#         shell=True)

#     seq_file_names = ['chr', 'start', 'end',
#                       'strand', 'label', 'seq']

#     train_seq = pd.read_csv(
#         f'single_prot_dbs/{cell_line}/{protein_name}/train_seq.tsv',
#                             sep='\t', names=seq_file_names)
#     val_seq = pd.read_csv(
#         f'single_prot_dbs/{cell_line}/{protein_name}/val_seq.tsv',
#                           sep='\t', names=seq_file_names)
#     test_seq = pd.read_csv(
#         f'single_prot_dbs/{cell_line}/{protein_name}/test_seq.tsv',
#                            sep='\t', names=seq_file_names)

#     pickling_batches(train_seq, 'train',
#                      f'single_prot_dbs/{cell_line}/{protein_name}',
#                      batch_size)
#     pickling_batches(val_seq, 'val',
#                      f'single_prot_dbs/{cell_line}/{protein_name}',
#                      batch_size)
#     pickling_batches(test_seq, 'test',
#                      f'single_prot_dbs/{cell_line}/{protein_name}',
#                      batch_size)
