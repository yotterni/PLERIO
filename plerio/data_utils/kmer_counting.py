import random
import numpy as np
import torch
import typing as tp


class KmerCounter:
    """
    A class to convert nucleic acid sequences to theis kmer
    representations.
    """
    def __init__(
        self, 
        ks: list[int], 
        stride: int | list[int] = 1,
        mode: str = 'soft', rna_alphabet: bool = False,
        torch_output: bool = False
    ) -> None:
        """
        :param ks: list of `k` indicates which k-mers are needed.
        :param stride: step for the sliding window.
        :param mode: `soft` means that class will skip the invalid kmers
         in the given string (e.g. that contains characters that are not.
         in the chosen alphabet). `hard` mode will force `KmerProcessor`
         to raise an error in such cases.
        :param rna_alphabet: which alphabet the `KmerProcessor` should use:
         'ATGC' (DNA) or 'AUGC' (RNA). Note that KmerProcessor with soft mode
         will successfully process RNA even with `rna_alphabet=False`, but it
         will just skip all the `U`-containing kmers which is probably not
         your objective. The same can be said for DNA, `rna_alphabet=False`
         and `T`, respectively. So it is highly recommended to use
         `KmerProcessor` with `mode='hard'` at least on the debugging stages
         of your project.
        :param torch_output: if `True`, the output will be a `torch.Tensor`
         instead of a `np.ndarray`.
        """
        assert all(isinstance(k, int) for k in ks), ('k for k-mers '
                                                     'should be integers')
        self.rna_alphabet = rna_alphabet
        self.ks_to_templates = dict(zip(ks,
                                        [self.create_template_kmer_dict_(k)
                                         for k in ks]))
        if isinstance(stride, int):
            self.ks_to_strides = dict(zip(ks, [stride] * len(ks)))
        else:
            assert len(stride) == len(ks), ('Stride must be single value or '
                                            'iterable with them same length as ks')
            self.ks_to_strides = dict(zip(ks, stride))

        assert mode in ['soft', 'hard'], 'Mode should be either "soft" or "hard"'
        self.mode = mode
        self.torch_output = torch_output

    def __call__(self, seq: str) -> np.ndarray[float] | torch.Tensor:
        """
        A high-level API for calling class insides. Calculates a
        k-mer representation of a nucleic acid sequence.
        :param seq: nucleic acid sequence.
        :return: k-mer count representation in format of
         np.ndarray or torch.Tensor.
        """
        return self.multi_k_calculator_(seq)

    def create_template_kmer_dict_(
        self, 
        k: int, 
        default_count: int = 0
    ) -> dict[str, int]:
        """
        Constructs a template k-mer dictionary. Each kmer count
        is set to the `default_count` (usually, 0).
        :param k: which k-mers are needed
        :param default_count: default value for each k-mer key in the dictionary
        :return: dict from kmer to its count, count will be set to `default_count`
        """
        if self.rna_alphabet:
            alphabet = 'AUGC'
        else:
            alphabet = 'ATGC'

        previous_list = list(alphabet)
        new_list = []
        for i in range(k - 1):
            for liter in previous_list:
                new_list.extend([liter + sym for sym in alphabet])
            previous_list = new_list
            new_list = []
        return dict.fromkeys(previous_list, default_count)


    def single_k_calculator_(
        self, 
        k_value: int, 
        seq: str
    ) -> np.ndarray[float]:
        """
        Calculates k-mer frequencies for a particular k.
        :param k_value: k for k-mers
        :param seq: sequence for representation calculation
        :return: np.ndarray, i-th coordinate is the count of
         i-th k-mer from all 4^k possible k-mers.
        """
        kmer_dict = self.ks_to_templates[k_value].copy()

        for i in range(0, len(seq) - k_value + 1,
                       self.ks_to_strides[k_value]):
            try:
                kmer_dict[seq[i:i + k_value]] += 1
            except KeyError:
                if self.mode == 'hard':
                    raise KeyError('Invalid character in the sequence!')
        return np.array(list(kmer_dict.values())).astype(float)


    def multi_k_calculator_(
        self, seq: str
    ) -> np.ndarray[float] | torch.Tensor:
        """
        Calls `self.single_k_calculator_` for each k passed in __init__.
        :param seq: nucleic acid sequence for representation calculation.
        :return: np.ndarray or torch.Tensor of concatenated representations.
        """
        single_k_representations = []
        for k in self.ks_to_templates:
            single_k_representations.append(self.single_k_calculator_(k, seq))

        concatenated_representations = np.hstack(single_k_representations)
        if self.torch_output:
            return torch.Tensor(concatenated_representations)
        else:
            return concatenated_representations

    def __str__(self) -> str:
        return f'KmerProcessor(ks={list(self.ks_to_templates)})'

    def __repr__(self) -> str:
        return self.__str__()
