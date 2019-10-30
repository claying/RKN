# Recurrent Kernel Networks for sequence data

This is a Pytorch implementation for reproducing the results of the paper

>Dexiong Chen, Laurent Jacob, Julien Mairal.
[Recurrent Kernel Networks][1]. In NeurIPS, 2019.


## Installation

We recommend users to use [anaconda][2] to install the following packages (link to [pytorch][3])

```
numpy
scipy
scikit-learn
pytorch=1.2.0
biopython
pandas
```

The code uses Pytorch just-in-time (JIT) to compile cpp and cuda extension code. 
Thus, you need to download [CUDA Toolkit 9.0][4] to some `$cuda_dir` and then run
```bash
export CUDA_HOME="$cuda_dir"
export TORCH_EXTENSIONS_DIR="$PWD/tmp"

export PYTHONPATH=$PWD:$PYTHONPATH
```
where the compiled extension files will be saved to `TORCH_EXTENSIONS_DIR`.

## Training models on SCOP 1.67 datasets

Download [SCOP 1.67 fold recognition datasets][5] and then do `mkdir data` and put the datasets to `./data/SCOP167-fold/`.

To train a model on a.102.1, with BLOSUM62 embedding, k=14, sigma=0.4 and max pooling, run
```bash
cd experiments
python train_scop.py --pooling max --embedding blosum62 --kmer-size 14 --alternating --sigma 0.4 --tfid 0
```

## Train multiclass model on SCOP 1.75 and test on SCOP 2.06

First download [SCOP 1.75 and SCOP 2.06][6] which were downloaded from [DeepSF][7] and preprocessed. 
Then unzip them respectively to `./data/SCOP175` and `./data/SCOP206`.

Then run
```bash
cd experiments
python train_scop175.py --alternating
```



[1]: https://arxiv.org/abs/1906.03200
[2]: https://anaconda.org
[3]: http://pytorch.org
[4]: https://developer.nvidia.com/cuda-90-download-archive
[5]: http://www.bioinf.jku.at/software/LSTM_protein/jLSTM_protein/datasets/SCOP167-fold.tar.bz2
[6]: http://pascal.inrialpes.fr/data2/dchen/data/SCOP175_206.zip
[7]: https://academic.oup.com/bioinformatics/article/34/8/1295/4708302
