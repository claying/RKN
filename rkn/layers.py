# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    PackedSequence, pack_padded_sequence, pad_packed_sequence)
import numpy as np

from . import ops
from .kernels import kernels
from .utils import normalize_, spherical_kmeans, spherical_kmeans2, get_mask, EPS
from .forget_rkn import (
    ForgetRKN, ForgetRKNPacked, ForgetRKNMaxPost,
    ForgetRKNPackedMaxPost, rkn_packed, rkn_forward,
    rkn_packed_max_lintrans, rkn_forward_max_lintrans
)
from .data.loader import BLOSUM62

from scipy import optimize
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin


class RKNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kmer_size=1,
                 gap_penalty=0.5, gap_penalty_trainable=False,
                 aggregation=False, la_feature=False,
                 kernel_func="exp", kernel_args=[0.5],
                 kernel_args_trainable=False,
                 additive=False, log_scale=False,
                 pooling='mean', agg_weight=1., unsup=False, pooling_arg=1e-03):
        super(RKNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kmer_size = kmer_size
        self.patch_dim = self.input_size * self.kmer_size
        self.unsup = unsup
        self.pooling_arg = pooling_arg

        # self.gap_penalty = gap_penalty
        self.gap_penalty_trainable = gap_penalty_trainable
        if gap_penalty_trainable:
            self.gap_penalty = nn.Parameter(torch.Tensor([gap_penalty]))
        else:
            self.register_buffer("gap_penalty", torch.Tensor([gap_penalty]))
        # sum the features
        self.aggregation = aggregation
        # compute LA feature
        self.la_feature = la_feature
        self.compute_h = aggregation and la_feature
        self.log_scale = log_scale
        self.additive = additive
        self.pooling = pooling

        self.kernel_args_trainable = kernel_args_trainable
        self.kernel_func = kernel_func
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == "exp":
            kernel_args = [1./kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = [kernel_arg / kmer_size for kernel_arg in kernel_args]
        if kernel_args_trainable:
            self.kernel_args = nn.ParameterList([nn.Parameter(torch.Tensor(
                [kernel_arg])) for kernel_arg in kernel_args])
        kernel_func = kernels[kernel_func]
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)

        self._need_lintrans_computed = True
        self.weight = nn.Parameter(
            torch.Tensor(hidden_size, kmer_size, input_size))

        self.register_buffer("lintrans",
                             torch.Tensor(hidden_size, hidden_size))

        #self.register_buffer("power", 1. / torch.arange(1., self.kmer_size + 1.))
        agg_weight = agg_weight ** torch.arange(0., kmer_size)
        agg_weight /= agg_weight.sum()
        self.register_buffer("agg_weight", agg_weight)

        self.reset_parameters()
        # self.normalize_()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                weight.data.uniform_(-stdv, stdv)
        self.normalize_()

    def train(self, mode=True):
        super(RKNLayer, self).train(mode)
        #if self.training is True:
        self._need_lintrans_computed = True

    def normalize_(self):
        if self.unsup:
            normalize_(self.weight.data.view(self.hidden_size, -1), dim=-1, c2=self.kmer_size)
        else:
            normalize_(self.weight.data, dim=-1)
        # normalize_(self.weight.data.view(self.hidden_size, -1), dim=-1, c2=self.kmer_size)

    def need_lintrans_computed(self, mode=True):
        self._need_lintrans_computed = mode

    def _compute_lintrans(self):
        if not self._need_lintrans_computed:
            return self.lintrans

        if self.kernel_func == 'identity':
            lintrans = torch.bmm(
                self.weight.permute(1, 0, 2), self.weight.permute(1, 2, 0))
        else:
            weight = self.weight
            lintrans = torch.bmm(
                weight.permute(1, 0, 2), weight.permute(1, 2, 0))
            lintrans = self.kappa(lintrans)
            # lintrans = lintrans * norm
        if self.log_scale and not self.additive:
            lintrans = lintrans.log()
            if self.aggregation:
                lintrans = lintrans.cumsum(dim=0)
            else:
                lintrans = lintrans.sum(dim=0, keepdim=True)
            lintrans = (lintrans.logsumexp(dim=0)) / self.kernel_args[0]# self.kmer_size
            lintrans = lintrans.exp()
        else:
            if self.aggregation:
                if self.additive:
                    lintrans = lintrans.sum(dim=0)
                else:
                    lintrans = lintrans.prod(dim=0)
            else:
                if self.additive:
                    lintrans = lintrans.sum(dim=0)
                else:
                    lintrans = lintrans.prod(dim=0)
        lintrans = ops.matrix_inverse_sqrt(lintrans)

        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data
        return lintrans

    def _conv_layer(self, input):
        """
        input: H x batch_size x input_size or all_length x input_size
        output: H x batch_size x hidden_size x kmer_size or
                all_length x hidden_size x kmer_size
        """
        if self.kernel_func == 'identity':
            out = torch.tensordot(input, self.weight, dims=([-1], [-1]))
        else:
            norm = input.norm(dim=-1, keepdim=True)
            norm = norm.unsqueeze(dim=-1)
            out = torch.tensordot(input, self.weight, dims=([-1], [-1]))
            out = out / norm.clamp(min=EPS)
            out = norm * self.kappa(out)
        return out

    def forward(self, input, hx=None):
        """
        input: H x batch_size x input_size
        output: H x batch_size x hidden_size
        """
        # normalize weight
        use_cuda = input.is_cuda
        self.normalize_()
        if self.gap_penalty.requires_grad:
            self.gap_penalty.data.clamp_(0, 0.99)

        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)

        if hx is None:
            hx = input.new_zeros(max_batch_size, self.hidden_size,
                                 self.kmer_size, requires_grad=False)

        input = self._conv_layer(input)
        
        lintrans = self._compute_lintrans()

        if use_cuda:
            forget = self.gap_penalty.expand_as(hx).contiguous()
            if batch_sizes is None:
                if self.pooling == 'max':
                    output, outputs, hx = ForgetRKNMaxPost.apply(input, forget, hx, self.la_feature, self.additive, lintrans)
                elif self.pooling == 'mean':
                    output, outputs, hx = ForgetRKN.apply(input, forget, hx, self.la_feature, self.additive)
                else:
                    output, outputs, hx = ForgetRKN.apply(input, forget, hx, False, self.additive)
            else:
                if self.pooling == 'max':
                    output, outputs, hx = ForgetRKNPackedMaxPost.apply(input, batch_sizes, forget, hx, self.la_feature, self.additive, lintrans)
                elif self.pooling == 'mean':
                    output, outputs, hx = ForgetRKNPacked.apply(input, batch_sizes, forget, hx, self.la_feature, self.additive)
                else:
                    output, outputs, hx = ForgetRKNPacked.apply(input, batch_sizes, forget, hx, False, self.additive)
        else:
            forget = self.gap_penalty
            if batch_sizes is None:
                if self.pooling == 'max':
                    output, outputs, hx = rkn_forward_max_lintrans(input, forget, hx, self.la_feature, self.additive, lintrans)
                elif self.pooling == 'mean':
                    output, outputs, hx = rkn_forward(input, forget, hx, self.la_feature, self.additive)
                else:
                    output, outputs, hx = rkn_forward(input, forget, hx, False, self.additive)
            else:
                if self.pooling == 'max':
                    output, outputs, hx = rkn_packed_max_lintrans(input, batch_sizes, forget, hx, self.la_feature, self.additive, lintrans)
                elif self.pooling == 'mean':
                    output, outputs, hx = rkn_packed(input, batch_sizes, forget, hx, self.la_feature, self.additive)
                else:
                    output, outputs, hx = rkn_packed(input, batch_sizes, forget, hx, False, self.additive)
        if self.aggregation:
            # print("aggregation")
            if self.pooling != 'max':
                output = F.linear(output.transpose(-1, -2), lintrans)
                outputs = F.linear(outputs.transpose(-1, -2), lintrans)
                output = output * self.agg_weight.view(1, -1, 1)
                outputs = outputs * self.agg_weight.view(1, 1, -1, 1)
            else:
                output = output * self.agg_weight.view(1, 1, -1)
                outputs = outputs * self.agg_weight.view(1, 1, 1, -1)
            output = output.view(list(output.shape[:-2]) + [-1])
            outputs = outputs.view(list(outputs.shape[:-2]) + [-1])
        else:
            output = output.select(dim=-1, index=-1)
            outputs = outputs.select(dim=-1, index=-1)
            if self.pooling != 'max':
                output = torch.mm(output, lintrans)
                outputs = F.linear(outputs, lintrans)

        if is_packed:
            outputs = PackedSequence(outputs, batch_sizes)

        if self.pooling == 'gmp':
            outputs, lengths = pad_packed_sequence(outputs)
            output = gmp_pooling(outputs, lengths, alpha=self.pooling_arg)

        return (output, outputs, hx)

    def extract_1d_patches(self, input):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, lengths = pad_packed_sequence(input)
            max_length = lengths[0]
        else:
            lengths = None
        output = input.unfold(0, self.kmer_size, 1).transpose(2, 3)
        output = output.contiguous()
        output = output.view(-1, self.kmer_size, self.input_size)

        if lengths is not None and (lengths != lengths[0]).any():
            mask = get_mask(lengths, max_length)
            mask = mask.unfold(0, self.kmer_size, 1).min(dim=-1)[0]
            mask = mask.view(-1) != 0
            output = output[mask]

        return output

    def sample_patches(self, input, n_sampling_patches=1000):
        """Sample patches from the given Tensor
        Args:
            input: H x batch_size x input_size or all_length x input_size
            n_sampling_patches (int): number of patches to sample
        Returns:
            patches:
            * x (input_size x kmer_size)
        """
        # print(torch.isnan(input[0]).any())
        patches = self.extract_1d_patches(input)
        # print(patches.shape)
        n_sampling_patches = min(patches.size(0), n_sampling_patches)

        indices = torch.randperm(patches.size(0))[:n_sampling_patches]
        patches = patches[indices]
        return patches

    def unsup_train(self, patches, init=None):
        """Unsupervised training for a CKN layer
        Args:
            patches: n x (input_size x kmer_size)
        Updates:
            filters: out_channels x (kmer_size x input_size)
        """
        print(patches.shape)
        if self.unsup:
            patches = patches.view(-1, self.patch_dim)
        normalize_(patches, dim=-1)
        if self.unsup:
            weight = spherical_kmeans(patches, self.hidden_size, init=init)
        else:
            weight = spherical_kmeans2(patches, self.hidden_size, init=init)
        # print(weight.norm(dim=-1))
        weight = weight.view_as(self.weight)
        self.weight.data.copy_(weight.data)
        # print(weight.shape)
        self.normalize_()
        self._need_lintrans_computed = True


class BioEmbedding(nn.Module):
    def __init__(self, num_embeddings, encoding='one_hot', pooling='max'):
        """Embedding layer for biosequences
        Args:
            num_embeddings (int): number of letters in alphabet
        """
        super(BioEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

        self.embedding = lambda x, weight: F.embedding(x, weight)
        # weight = self._make_weight()
        if encoding == 'blosum62':
            if pooling == 'max':
                weight = np.sqrt(BLOSUM62)
            else:
                weight = BLOSUM62 - BLOSUM62.mean(axis=1, keepdims=True)
                weight = weight / np.linalg.norm(weight, axis=1, keepdims=True).clip(1e-6)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = self._make_weight()
        self.register_buffer("weight", weight)

    def _make_weight(self):
        weight = torch.zeros(self.num_embeddings + 1, self.num_embeddings)
        weight[0] = 1./self.num_embeddings
        weight[1:] = torch.diag(torch.ones(self.num_embeddings))
        return weight

    def forward(self, input, lengths):
        """
        Args:
            input: LongTensor of indices batch_size x H
            output: H x batch_size x embed_size
        """
        output = self.embedding(input, self.weight)
        output = output.transpose(0, 1).contiguous()
        return pack_padded_sequence(output, lengths)


class Linear(nn.Linear, LinearModel, LinearClassifierMixin):
    def __init__(self, in_features, out_features, alpha=0.0, fit_bias=True,
                 penalty="l2"):
        super(Linear, self).__init__(in_features, out_features, fit_bias)
        self.alpha = alpha
        self.fit_bias = fit_bias
        self.penalty = penalty

    def forward(self, input, proba=False):
        out = super(Linear, self).forward(input)
        if proba:
            return out.sigmoid()
        return out

    def fit(self, x, y, criterion=None):
        use_cuda = self.weight.data.is_cuda
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        reduction = criterion.reduction
        criterion.reduction = 'sum'
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        def eval_loss(w):
            w = w.reshape((self.out_features, -1))
            if self.weight.grad is not None:
                self.weight.grad = None
            if self.bias is None:
                self.weight.data.copy_(torch.from_numpy(w))
            else:
                if self.bias.grad is not None:
                    self.bias.grad = None
                self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
                self.bias.data.copy_(torch.from_numpy(w[:, -1]))
            y_pred = self(x).squeeze_(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            if self.alpha != 0.0:
                if self.penalty == "l2":
                    penalty = 0.5 * self.alpha * torch.norm(self.weight)**2
                elif self.penalty == "l1":
                    penalty = self.alpha * torch.norm(self.weight, p=1)
                    penalty.backward()
                loss = loss + penalty
            return loss.item()

        def eval_grad(w):
            dw = self.weight.grad.data
            if self.alpha != 0.0:
                if self.penalty == "l2":
                    dw.add_(self.alpha, self.weight.data)
            if self.bias is not None:
                db = self.bias.grad.data
                dw = torch.cat((dw, db.view(-1, 1)), dim=1)
            return dw.cpu().numpy().ravel().astype("float64")

        w_init = self.weight.data
        if self.bias is not None:
            w_init = torch.cat((w_init, self.bias.data.view(-1, 1)), dim=1)
        w_init = w_init.cpu().numpy().astype("float64")

        w = optimize.fmin_l_bfgs_b(
            eval_loss, w_init, fprime=eval_grad, maxiter=100, disp=0)
        if isinstance(w, tuple):
            w = w[0]

        w = w.reshape((self.out_features, -1))
        self.weight.grad.data.zero_()
        if self.bias is None:
            self.weight.data.copy_(torch.from_numpy(w))
        else:
            self.bias.grad.data.zero_()
            self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
            self.bias.data.copy_(torch.from_numpy(w[:, -1]))
        criterion.reduction = reduction

    def decision_function(self, x):
        x = torch.from_numpy(x)
        return self(x).data.numpy().ravel()

    def predict(self, x):
        return self.decision_function(x)

    def predict_proba(self, x):
        return self._predict_proba_lr(x)

    @property
    def coef_(self):
        return self.weight.data.numpy()

    @property
    def intercept_(self):
        return self.bias.data.numpy()


def gmp_pooling(x, lengths=None, alpha=1e-3):
    x = x.transpose(0, 1)
    xxt = torch.bmm(x.transpose(1, 2), x)
    xxt.diagonal(dim1=1, dim2=2)[:] += alpha
    eye = xxt.new_ones(xxt.size(-1)).diag().expand_as(xxt)
    xxt, _ = torch.gesv(eye, xxt)
    x = torch.bmm(x, xxt)
    return x.sum(dim=1)
