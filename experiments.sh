#!/usr/bin/env bash

#
# PCA
#
python benchmark.py -d cdr-supervised -c configs/pca.json -g configs/top5/cdr-supervised_pca.json -E 200 > cdr-supervised_pca.log 2> cdr-supervised_pca.err
python benchmark.py -d cdr-dp -c configs/pca.json -g configs/top5/cdr-supervised_pca.json -E 200 > cdr-dp_pca.log 2> cdr-dp_pca.err
python benchmark.py -d spouse-supervised -c configs/pca.json -g configs/top5/spouse-supervised_pca.json -E 200 > spouse-supervised_pca.log 2> spouse-supervised_pca.err
python benchmark.py -d spouse-dp -c configs/pca.json -g configs/top5/spouse-dp_pca.json -E 200 > spouse-dp_pca.log 2> spouse-dp_pca.err

#
# PCA + Exp. Decay Kernel
#
python benchmark.py -d cdr-supervised -c configs/pca-kernel.json -g configs/top5/cdr-supervised_pca-kernel.json -E 200 > cdr-supervised_pca-kernel.log 2> cdr-supervised_pca-kernel.err
python benchmark.py -d cdr-dp -c configs/pca-kernel.json -g configs/top5/cdr-supervised_pca-kernel.json -E 200  > cdr-dp_pca-kernel.log 2> cdr-dp_pca-kernel.err
python benchmark.py -d spouse-supervised -c configs/pca-kernel.json -g configs/top5/spouse-supervised_pca-kernel.json -E 200 > spouse-supervised_pca-kernel.log 2> spouse-supervised_pca-kernel.err
python benchmark.py -d spouse-dp -c configs/pca-kernel.json -g configs/top5/spouse-dp_pca-kernel.json -E 200 > spouse-dp_pca-kernel.log 2> spouse-dp_pca-kernel.err

#
# LSTM
#
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d cdr-supervised -c configs/lstm.json -g configs/top5/cdr-supervised_lstm.json -E 200 -H GPU > cdr-supervised_lstm.log 2> cdr-supervised_lstm.err
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d cdr-dp -c configs/lstm.json -g configs/top5/cdr-dp_lstm.json -E 200 -H GPU > cdr-dp_lstm.log 2> cdr-dp_lstm.err
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d spouse-supervised -c configs/lstm.json -g configs/top5/spouse-supervised_lstm.json -E 200 -H GPU > spouse-supervised_lstm.log 2> spouse-supervised_lstm.err
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d spouse-dp -c configs/lstm.json -g configs/top5/spouse-dp_lstm.json -E 200 -H GPU > spouse-dp_lstm.log 2> spouse-dp_lstm.err

#
# LSTM + Attention
#
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d cdr-supervised -c configs/lstm.json -p attention=True -g configs/top5/cdr-supervised_lstm.json -E 200 -H GPU > cdr-supervised_lstm-attn.log 2> cdr-supervised_lstm-attn.err
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d cdr-dp -c configs/lstm.json -p attention=True -g configs/top5/cdr-dp_lstm.json -E 200 -H GPU > cdr-dp_lstm-attn.log 2> cdr-dp_lstm-attn.err
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d spouse-supervised -c configs/lstm.json -p attention=True -g configs/top5/spouse-supervised_lstm.json -E 200 -H GPU > spouse-supervised_lstm-attn.log 2> spouse-supervised_lstm-attn.err
CUDA_VISIBLE_DEVICES=0 python benchmark.py -d spouse-dp -c configs/lstm.json -p attention=True -g configs/top5/spouse-dp_lstm.json -E 200 -H GPU > spouse-dp_lstm-attn.log 2> spouse-dp_lstm-attn.err
