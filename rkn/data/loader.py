# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from collections import defaultdict
from sklearn.model_selection import train_test_split

from .data_helper import pad_sequences, TensorDataset, augment

if sys.version_info < (3,):
    import string
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

ALPHABETS = {
    'DNA': (
        'ACGT',
        '\x01\x02\x03\x04'
    ),
    'PROTEIN': (
        'ARNDCQEGHILKMFPSTWYV',
        '\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14'
    ),
}

AMBIGUOUS = {
    'DNA': ('N', '\x00'),
    'PROTEIN': ('XBZJUO', '\x00' * 6),
}

BLOSUM62 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
    [29,   3,   2,   2,   2,   2,   4,   7,   1,   4,   5,   4,   1,   2,   2,   8,   4,   0,   1,   6],
    [4,  34,   3,   3,   0,   4,   5,   3,   2,   2,   4,  12,   1,   1,   1,   4,   3,   0,   1,   3],
    [4,   4,  31,   8,   0,   3,   4,   6,   3,   2,   3,   5,   1,   1,   2,   6,   4,   0,   1,   2],
    [4,   2,   6,  39,   0,   2,   9,   4,   1,   2,   2,   4,   0,   1,   2,   5,   3,   0,   1,   2],
    [6,   1,   1,   1,  48,   1,   1,   3,   0,   4,   6,   2,   1,   2,   1,   4,   3,   0,   1,   5],
    [5,   7,   4,   4,   0,  21,  10,   4,   2,   2,   4,   9,   2,   1,   2,   5,   4,   0,   2,   3],
    [5,   4,   4,   9,   0,   6,  29,   3,   2,   2,   3,   7,   1,   1,   2,   5,   3,   0,   1,   3],
    [7,   2,   3,   3,   1,   1,   2,  51,   1,   1,   2,   3,   0,   1,   1,   5,   2,   0,   1,   2],
    [4,   4,   5,   3,   0,   3,   5,   3,  35,   2,   3,   4,   1,   3,   1,   4,   2,   0,   5,   2],
    [4,   1,   1,   1,   1,   1,   1,   2,   0,  27,  16,   2,   3,   4,   1,   2,   3,   0,   2,  17],
    [4,   2,   1,   1,   1,   1,   2,   2,   1,  11,  37,   2,   4,   5,   1,   2,   3,   0,   2,   9],
    [5,  10,   4,   4,   0,   5,   7,   4,   2,   2,   4,  27,   1,   1,   2,   5,   3,   0,   1,   3],
    [5,   3,   2,   2,   1,   2,   2,   2,   1,  10,  19,   3,  16,   4,   1,   3,   4,   0,   2,   9],
    [3,   1,   1,   1,   1,   1,   1,   2,   1,   6,  11,   1,   2,  38,   1,   2,   2,   1,   8,   5],
    [5,   2,   2,   3,   1,   2,   3,   3,   1,   2,   3,   4,   1,   1,  49,   4,   3,   0,   1,   3],
    [10,   4,   5,   4,   1,   3,   5,   6,   1,   2,   4,   5,   1,   2,   2,  21,   8,   0,   1,   4],
    [7,   3,   4,   3,   1,   2,   3,   4,   1,   5,   6,   4,   1,   2,   2,   9,  24,   0,   1,   7],
    [3,   2,   1,   1,   0,   1,   2,   3,   1,   3,   5,   2,   1,   6,   0,   2,   2,  49,   6,   3],
    [4,   2,   2,   1,   0,   2,   2,   2,   4,   4,   6,   3,   1,  13,   1,   3,   2,   2,  31,   4],
    [6,   2,   1,   1,   1,   1,   2,   2,   0,  16,  13,   2,   3,   3,   1,   3,   4,   0,   2,  26]])
BLOSUM62 = BLOSUM62 / BLOSUM62.sum(axis=1, keepdims=True)


class SeqLoader(object):
    def __init__(self, alphabet='DNA'):
        self.alphabet, self.code = ALPHABETS[alphabet]
        alpha_ambi, code_ambi = AMBIGUOUS[alphabet]
        self.translator = maketrans(
            alpha_ambi + self.alphabet, code_ambi + self.code)
        self.alpha_nb = len(self.alphabet)

    def pad_seq(self, seq):
        seq = pad_sequences(seq, pre_padding=self.pre_padding,
                            maxlen=self.maxlen, padding='post',
                            truncating='post', dtype='int64')
        self.maxlen = seq.shape[1]
        return seq

    def seq2index(self, seq):
        seq = seq.translate(self.translator)
        seq = np.fromstring(seq, dtype='uint8')
        return seq.astype('int64')

    def get_ids(self, ids=None):
        pass

    def get_tensor(self, dataid, split='train', noise=0.0, fixed_noise=0.0,
                   val_split=0.25, top=False, generate_neg=True, aug_quant=10,
                   n_top=500, return_len=True):
        df = self.load_data(dataid, split)
        if fixed_noise > 0.:
            df = augment(df, noise=fixed_noise, quantity=aug_quant,
                         max_index=self.alpha_nb)
        if split == 'train' and generate_neg and hasattr(self, "aug_neg"):
            df = self.aug_neg(df)
        X, y = df['seq_index'], df['y'].values
        lengths = np.asarray([np.clip(len(s) + 2 * self.pre_padding,
                             a_min=0, a_max=self.maxlen) for s in X])
        X = self.pad_seq(X)
        if top:
            X, _, y, _, lengths, _ = train_test_split(
                X, y, lengths, stratify=y, train_size=n_top, random_state=1)
        if split == 'train' and val_split > 0:
            X, X_val, y, y_val, lengths, lengths_val = train_test_split(
                X, y, lengths, test_size=val_split, stratify=y, random_state=1)
            X, y = torch.from_numpy(X), torch.from_numpy(y)
            lengths = torch.from_numpy(lengths)
            X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
            lengths_val = torch.from_numpy(lengths_val)
            train_dset = TensorDataset(
                X, y, lengths, noise=noise, max_index=self.alpha_nb, return_len=return_len)
            val_dset = TensorDataset(X_val, y_val, lengths_val, max_index=self.alpha_nb,
                                     return_len=return_len)
            return train_dset, val_dset
        X, y = torch.from_numpy(X), torch.from_numpy(y)
        lengths = torch.from_numpy(lengths)
        return TensorDataset(X, y, lengths, noise=noise, max_index=self.alpha_nb,
                             return_len=return_len)


class SCOPLoader(SeqLoader):
    def __init__(self, datadir='data/SCOP', ext='fasta', maxlen=None,
                 pre_padding=0):
        super(SCOPLoader, self).__init__('PROTEIN')
        self.datadir = datadir
        self.ext = ext
        self.pre_padding = pre_padding
        self.maxlen = maxlen
        self.filename_tp = datadir + '/{}-{}.{}.' + ext

    def get_ids(self, ids=None):
        names = sorted([".".join(filename.split('.')[1:-1])
                       for filename in os.listdir(self.datadir)
                       if filename.startswith('pos-train')])
        if ids is not None and ids != []:
            names = [names[index] for index in ids if index < len(names)]
        return names

    def get_nb_fasta(self, filename):
        records = SeqIO.parse(filename, self.ext)
        return sum([1 for r in records])

    def get_nb(self, dataid, split='train'):
        pos = self.filename_tp.format('pos', split, dataid)
        neg = self.filename_tp.format('neg', split, dataid)
        return self.get_nb_fasta(pos) + self.get_nb_fasta(neg)

    def load_data(self, dataid, split='train'):
        pos = self.filename_tp.format('pos', split, dataid)
        neg = self.filename_tp.format('neg', split, dataid)
        table = defaultdict(list)
        for y, filename in enumerate([neg, pos]):
            for record in SeqIO.parse(filename, self.ext):
                seq = str(record.seq).upper()
                name = record.id
                table['Seq'].append(seq)
                table['seq_index'].append(self.seq2index(seq))
                table['name'].append(name)
                table['y'].append(y)
        df = pd.DataFrame(table)
        return df
