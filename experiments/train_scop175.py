import os
import argparse

from rkn.data.loader_scop import load_data
from rkn.models import RKN
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
import torch.optim as optim
import numpy as np

import copy
from timeit import default_timer as timer

train_name = 'SCOP175'
test_name = 'SCOP206'
traindir = '../data/{}'.format(train_name)
testdir = '../data/{}'.format(test_name)
train_list = 'Traindata'
# val_list = 'Testdata'
val_ref = 'Testdata_id{}againstTrain'
val_list = [95, 70, 40, 25]
test_list = 'SCOP206'


def load_args():
    parser = argparse.ArgumentParser(
        description="RKN for SCOP 1.75 and SCOP 2.06 experiments")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        "--tfid", metavar="tfid", dest="tfid", default=[], nargs='*',
        type=int, help="tfids to generate experiments")
    parser.add_argument(
        '--batch-size', type=int, default=128, metavar='M',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--epochs', type=int, default=100, metavar='N',
        help='number of epochs to train (default: 100)')
    parser.add_argument(
        "--hidden-size", default=[128], nargs='+', type=int,
        help="hidden size for each layer (default [128])")
    parser.add_argument(
        "--kmer-size", default=[10], nargs='+', type=int,
        help="kmer size for each layer (default: [10])")
    parser.add_argument(
        "--gap-penalty", default=[0.4], nargs='+', type=float,
        help="gap penalty for each layer (default: [0.4])")
    parser.add_argument(
        "--sigma", default=[0.4],
        nargs='+', type=float, help="sigma for each layer (default: [0.4])")
    parser.add_argument(
        "--gap-penalty-trainable", action='store_true',
        help="train gap penalty or not")
    parser.add_argument(
        "--aggregation", action='store_true',
        help='aggregate features or not')
    parser.add_argument(
        "--la-feature", default=None, type=bool,
        help='use LA features or not')
    parser.add_argument(
        "--sampling-patches", dest="n_sampling_patches", default=300000,
        type=int, help="number of sampled patches (default: 300000)")
    parser.add_argument(
        "--penalty", metavar="penal", dest="penalty", default='l2',
        type=str, choices=['l2', 'l1'],
        help="regularization used in the last layer (default: l2)")
    parser.add_argument(
        "--outdir", metavar="outdir", dest="outdir",
        default='', type=str, help="output path(default: '')")
    parser.add_argument(
        "--regularization", type=float, default=0.1,
        help="regularization parameter for RKN (default: 0.1)")
    parser.add_argument("--alternating", action='store_true',
        help="alternating training or not (default: False)")
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
    parser.add_argument(
        "--use-cuda", action='store_true', default=False,
        help="use gpu (default: False)")
    parser.add_argument(
        "--pooling", type=str, default='mean',
        choices=['mean', 'max', 'gmp'], help='pooling method [mean, max, gmp]')
    parser.add_argument(
        "--agg-weight", type=float, default=2.,
        help='weights for aggregating kernels with different lengths')
    parser.add_argument(
        "--pooling-arg", default=0.001, type=float,
        help='arg for global pooling (default: 0.001)')

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.use_cuda = True
    # check shape
    args.n_layers = len(args.hidden_size)

    args.save_logs = False
    if args.outdir != "":
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(test_name)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outdir = outdir + "/{}".format(args.pooling)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        if args.alternating:
            outdir = outdir + "/alternating"
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except:
                    pass
        outdir = outdir+'/{}_{}_{}_{}_{}_{}'.format(
            args.n_layers, args.hidden_size, args.kmer_size, args.gap_penalty, args.sigma,
            args.regularization)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        args.outdir = outdir

    return args


class BioEmbedding(nn.Module):
    def forward(self, input, lengths):
        output = input.transpose(0, 1).contiguous()
        return pack_padded_sequence(output, lengths)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.gap_penalty = args.gap_penalty[0]
        self.embed_layer = BioEmbedding()
        self.n_class = args.n_class
        self.rkn = RKN(
            45, args.n_class, args.hidden_size, args.kmer_size, args.gap_penalty,
            args.gap_penalty_trainable, args.aggregation, args.la_feature,
            kernel_args_list=args.sigma, alpha=args.regularization,
            penalty=args.penalty, log_scale=False, pooling=args.pooling,
            agg_weight=args.agg_weight, pooling_arg=args.pooling_arg)

    def representation(self, input, lengths):
        output = self.embed_layer(input, lengths)
        return self.rkn.representation(output)

    def representation_at(self, input, lengths, i):
        output = self.embed_layer(input, lengths)
        return self.rkn.representation_at(output, i)

    def forward(self, input, lengths, proba=False):
        output = self.embed_layer(input, lengths)
        return self.rkn(output, proba=proba)[0]

    def unsup_train_rkn(self, data_loader, n_sampling_patches=100000,
                        init=None, use_cuda=False):
        self.train(False)
        if use_cuda:
            self.cuda()

        for i, rkn_layer in enumerate(self.rkn.rkn_model):
            print("Training layer {}".format(i))
            n_patches = 0
            try:
                n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader)
            except:
                n_patches_per_batch = 1000
            # patches = torch.Tensor(n_sampling_patches, rkn_layer.patch_dim)
            patches = torch.Tensor(n_sampling_patches, rkn_layer.kmer_size, rkn_layer.input_size)
            if use_cuda:
                patches = patches.cuda()

            for data, _, lengths in data_loader:
                if n_patches >= n_sampling_patches:
                    continue
                lengths, indices = lengths.sort(descending=True)
                data = data[indices]
                if use_cuda:
                    data = data.cuda()
                    lengths = lengths.cuda()
                with torch.no_grad():
                    data = self.representation_at(data, lengths, i)
                    data_patches = rkn_layer.sample_patches(
                        data, n_patches_per_batch)
                size = data_patches.size(0)
                if n_patches + size > n_sampling_patches:
                    size = n_sampling_patches - n_patches
                    data_patches = data_patches[:size]
                patches[n_patches: n_patches + size] = data_patches
                n_patches += size

            print("total number of patches: {}".format(n_patches))
            patches = patches[:n_patches]
            rkn_layer.unsup_train(patches, init=init)

    def unsup_train_classifier(self, train_loader, criterion, use_cuda=True):
        encoded_train, encoded_target = self.predict(
            train_loader, only_representation=True, use_cuda=use_cuda)
        self.rkn.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False,
                proba=False, use_cuda=False):
        self.eval()
        if use_cuda:
            self.cuda()
        n_samples = len(data_loader.dataset)
        target_output = torch.LongTensor(n_samples)
        batch_start = 0
        for i, (data, target, lengths, *_) in enumerate(data_loader):
            lengths, indices = lengths.sort(descending=True)
            data = data[indices]
            target = target[indices]
            batch_size = data.shape[0]
            if use_cuda:
                data = data.cuda()
                lengths = lengths.cuda()
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data, lengths).data.cpu()
                else:
                    batch_out = self(data, lengths, proba).data.cpu()
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
            output[batch_start:batch_start+batch_size] = batch_out
            target_output[batch_start:batch_start+batch_size] = target
            batch_start += batch_size
        output.squeeze_(-1)
        return output, target_output

    def sup_train(self, data_loader, criterion, optimizer, lr_scheduler=None,
                  epochs=100, early_stop=True, alternating=False,
                  n_sampling_patches=300000, unsup_init='kmeans++',
                  use_cuda=False):
        phases = ['train', 'val']

        if use_cuda:
            self.cuda()

        self.unsup_train_rkn(data_loader['init'], n_sampling_patches,
                                 init=unsup_init, use_cuda=use_cuda)

        best_loss = float('inf')
        best_acc = 0
        best_epoch = 0
        epoch_loss = None

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)
            if alternating:
                self.unsup_train_classifier(data_loader['train'], criterion, use_cuda)

            if lr_scheduler is not None:
                # lr_scheduler.step()
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    if epoch_loss is not None:
                        lr_scheduler.step(epoch_loss)
                else:
                    lr_scheduler.step()
                print("current LR: {}".format(
                      optimizer.param_groups[0]['lr']))
            print(self.rkn.rkn_model[0].gap_penalty.item())

            for phase in phases:
                if phase not in data_loader:
                    continue
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                train_loader = data_loader[phase]
                tic = timer()
                if isinstance(train_loader, list):
                    epoch_loss = []
                    epoch_acc = []
                    for ids, train_l in train_loader:
                        e_loss, e_acc = self.one_step(
                            phase, train_l, optimizer, criterion, use_cuda)
                        epoch_loss.append(e_loss)
                        epoch_acc.append(e_acc)
                else:
                    epoch_loss, epoch_acc = self.one_step(
                        phase, train_loader, optimizer, criterion, use_cuda)
                toc = timer()

                if isinstance(epoch_loss, list):
                    epoch_loss = np.mean(epoch_loss)
                    epoch_acc = np.mean(epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.2f}s'.format(
                    phase, epoch_loss, epoch_acc, toc - tic))

                if (phase == 'val') and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch + 1
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())

            print()

        print('Best epoch: {}'.format(best_epoch))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best val loss: {:4f}'.format(best_loss))
        if early_stop:
            self.load_state_dict(best_weights)

        return best_loss, best_acc, best_epoch

    def one_step(self, phase, train_loader, optimizer, criterion, use_cuda):
        running_loss = 0.0
        running_acc = 0.0
        # tic = timer()
        # train_loader = data_loader[phase]

        for data, target, lengths in train_loader:
            size = data.size(0)
            with torch.no_grad():
                lengths, indices = lengths.sort(descending=True)
                data = data[indices]
                target = target[indices]
                # target = target.float()
            if use_cuda:
                data = data.cuda()
                lengths = lengths.cuda()
                target = target.cuda()

            if phase == "train":
                optimizer.zero_grad()
                output = self.forward(data, lengths)
                loss = criterion(output, target)
                pred = output.data.argmax(dim=1)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    output = self.forward(data, lengths)#.view(-1)
                    loss = criterion(output, target)
                    pred = output.data.argmax(dim=1)

            running_loss += loss.item() * size
            running_acc += torch.sum(pred == target.data).item()
            # print(out)

        # toc = timer()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_acc / len(train_loader.dataset)
        return epoch_loss, epoch_acc



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def main():
    args = load_args()
    print(args)
    args.n_class = 1195
    pre_padding = (args.kmer_size[0] - 1) // 2
    maxlen=None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_dset = load_data(traindir, train_list, maxlen=maxlen, pre_padding=pre_padding)
    if isinstance(val_list, list):
        val_dset = [
        load_data(traindir, val_ref.format(val_l), maxlen=maxlen, pre_padding=pre_padding) for val_l in val_list]
    else:
        val_dset = load_data(traindir, val_list, maxlen=maxlen, pre_padding=pre_padding)

    loader_args = {}
    if args.use_cuda:
        loader_args = {'num_workers': 1, 'pin_memory': True}

    init_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, **loader_args)
    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, **loader_args)

    if isinstance(val_list, list):
        val_loader = [(val_l, DataLoader(
            val_d, batch_size=args.batch_size, shuffle=False, **loader_args)) for val_l, val_d in zip(val_list, val_dset)]
    else:
        val_loader = DataLoader(
            val_dset, batch_size=args.batch_size, shuffle=False, **loader_args)

    data_loader = {'train': train_loader, 'val': val_loader, 'init': init_loader}

    model = Net(args)

    if args.alternating:
        optimizer = optim.Adam(model.rkn.rkn_model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam([
            {'params': model.rkn.rkn_model.parameters()},
            {'params': model.rkn.classifier.parameters(), 'weight_decay': args.regularization}
            ], lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    lr_scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-4)

    print(model.rkn.rkn_model[0].gap_penalty)

    print("Start training...")
    tic = timer()
    val_loss, val_acc, val_epoch = model.sup_train(data_loader, criterion, optimizer, lr_scheduler,
        epochs=args.epochs, alternating=args.alternating, use_cuda=args.use_cuda)
    toc = timer()
    training_time = (toc - tic) / 60


    test_dset = load_data(testdir, test_list, maxlen=maxlen, pre_padding=pre_padding,
                          label_file='fold_label_relation2.txt')
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False)
    y_pred, y_true = model.predict(
        test_loader, proba=False, use_cuda=args.use_cuda)

    scores = accuracy(y_pred, y_true, (1, 5, 10, 20))
    print(scores)
    test_indices = np.load(testdir + '/test_indices.npz')
    stratified_scores = {}
    for idx_key in test_indices:
        idx = test_indices[idx_key]
        stratified_scores[idx_key] = accuracy(y_pred[idx], y_true[idx], (1, 5, 10, 20))
    print(stratified_scores)

    if args.save_logs:
        import pandas as pd
        scores = {
            'top1': scores[0],
            'top5': scores[1],
            'top10': scores[2],
            'top20': scores[3],
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_epoch': val_epoch
        }

        scores = pd.DataFrame.from_dict(scores, orient='index')
        scores.to_csv(args.outdir + '/metric.csv',
                  header=['value'], index_label='name')
        for idx in stratified_scores:
            s = stratified_scores[idx]
            s = {'top1': s[0], 'top5': s[1], 'top10': s[2], 'top20': s[3]}
            s = pd.DataFrame.from_dict(s, orient='index')
            s.to_csv(args.outdir + '/{}.csv'.format(idx),
                  header=['value'], index_label='name')

        np.save(args.outdir + "/predict", y_pred.numpy())
        torch.save(
            {'args': args,
             'state_dict': model.state_dict()},
            args.outdir + '/model.pkl')


if __name__ == "__main__":
    main()
