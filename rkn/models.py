# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from .layers import RKNLayer, Linear


class RKNSequential(nn.Module):
    def __init__(self, input_size, hidden_sizes, kmer_sizes,
                 gap_penalties=None, gap_penalty_trainable=False,
                 aggregations=None, la_features=None,
                 kernel_funcs=None, kernel_args_list=None,
                 kernel_args_trainable=False, **kwargs):

        super(RKNSequential, self).__init__()
        self.input_size = input_size
        self.output_size = hidden_sizes[-1]
        self.hidden_sizes = hidden_sizes
        self.kmer_sizes = kmer_sizes
        self.n_layers = len(hidden_sizes)

        rkn_layers = []

        for i in range(self.n_layers):
            if gap_penalties is None:
                gap_penalty = 0.5
            elif isinstance(gap_penalties, list):
                gap_penalty = gap_penalties[i]
            else:
                gap_penalty = gap_penalties

            if aggregations is None:
                aggregation = False
            elif isinstance(aggregations, list):
                aggregation = aggregations[i]
            else:
                aggregation = aggregations

            if la_features is None:
                la_feature = False
                if i == self.n_layers - 1:
                    la_feature = True
            elif isinstance(la_features, list):
                la_feature = la_features[i]
            else:
                la_feature = la_features

            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.5
            else:
                kernel_args = kernel_args_list[i]

            rkn_layer = RKNLayer(input_size, hidden_sizes[i], kmer_sizes[i],
                                 gap_penalty, gap_penalty_trainable,
                                 aggregation, la_feature, kernel_func,
                                 kernel_args, kernel_args_trainable, **kwargs)

            rkn_layers.append(rkn_layer)
            input_size = hidden_sizes[i]

        self.rkn_layers = nn.ModuleList(rkn_layers)

    def __getitem__(self, idx):
        return self.rkn_layers[idx]

    def __len__(self):
        return len(self.rkn_layers)

    def __iter__(self):
        return iter(self.rkn_layers._modules.values())

    def forward(self, input, hidden=None):
        next_hidden = []
        for i, layer in enumerate(self.rkn_layers):
            output, input, next_hx = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(next_hx)
        return output, input, next_hidden

    def forward_at(self, input, i=0, hidden=None):
        return self.rkn_layers[i](input, None if hidden is None else hidden[i])

    def representation(self, input, n=0, hidden=None):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            _, input, _ = self.forward_at(input, i, hidden)
        return input

    def normalize_(self):
        for module in self.rkn_layers:
            module.normalize_()

    def need_lintrans_computed(self, mode=True):
        for module in self.rkn_layers:
            module.need_lintrans_computed(mode)


class RKN(nn.Module):
    def __init__(self, input_size, n_classes, hidden_sizes, kmer_sizes,
                 gap_penalties=None, gap_penalty_trainable=False,
                 aggregations=None, la_features=None,
                 kernel_funcs=None, kernel_args_list=None,
                 kernel_args_trainable=False, alpha=0., fit_bias=True,
                 penalty='l2', **kwargs):
        super(RKN, self).__init__()
        self.rkn_model = RKNSequential(input_size, hidden_sizes, kmer_sizes,
                                       gap_penalties, gap_penalty_trainable,
                                       aggregations, la_features, kernel_funcs,
                                       kernel_args_list, kernel_args_trainable,
                                       **kwargs)
        if aggregations:
            self.output_size = hidden_sizes[-1] * kmer_sizes[-1]
        else:
            self.output_size = hidden_sizes[-1]
        self.n_classes = n_classes
        self.classifier = Linear(self.output_size, n_classes, alpha, fit_bias,
                                 penalty)

    def representation(self, input, hidden=None):
        return self.rkn_model(input, hidden)[0]

    def representation_at(self, input, i, hidden=None):
        return self.rkn_model.representation(input, i, hidden)

    def forward(self, input, hidden=None, proba=False, predict_all=False, dropout=None):
        output, outputs, hidden = self.rkn_model(input, hidden)
        if predict_all:
            output = outputs.view(-1, self.output_size)
        if dropout is not None:
            output = dropout(output)
        return self.classifier(output, proba), hidden
