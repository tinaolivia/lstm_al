#!/usr/bin/env python3
import torch
import argparse
import datetime
import pandas as pd
import sys

import al
import helpers

import torchtext.data as data
from pathlib import Path
from fastai.text import *

parser = argparse.ArgumentParser(description='LSTM language model and text classifier')
# general
parser.add_argument('-bs', type=int, default=32, help='batch size [default: 32]')
parser.add_argument('-lr', type=float, default=1e-2, help='maximum learning rate [default: 3e-2]')
parser.add_argument('-momentum', type=tuple, default=(0.8, 0.7), help='tuple of momentum for optimization [default: (0.8, 0.7)]')
parser.add_argument('-epochs', type=int, default=50, help='maximum number of epochs [default: 50]')
parser.add_argument('-earlystop', type=float, default=0.01, help='early stopping criterion [default: 0.01]')
parser.add_argument('-save-dir', type=str, default='results', help='where to store model resulst [default: results]')
parser.add_argument('-num-avg', type=int, default=10, help='number of runs to average over [default: 10]')
# device
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# model
parser.add_argument('-model', type=str, default='enc', help='name for storing language model [default: enc]')
parser.add_argument('-pretrained', type=bool, default=True, help='using pretrained model [default: True]')
# data
parser.add_argument('-path', type=str, default='data', help='path to data [default: data/dataset (e.g. data/imdb/)]')
parser.add_argument('-dataset', type=str, default='imdb', choices=['imdb', 'ag'], help='dataset [default: imdb]')
parser.add_argument('-text-first', type=bool, default=True, help='whether text column (True) or label column (False) is first [default: True]')
# active learning
parser.add_argument('-method', type=str, default=None, help='active learning method [default: None]')
parser.add_argument('-rounds', type=int, default=100, help='number of active learning loops [default:100]')
parser.add_argument('-inc', type=int, default=1, help='number of instances added at each active learning loop [default: 1]')
parser.add_argument('-cluster', type=bool, default=False, help='whether to cluster unlabeled data before active learning [default: False]')
# defining parser
args = parser.parse_args()

# defining text and labeld fields
print('Creating fields \n')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

# defining additional or final arguments
print('Defining additional/final arguments. \n')
args.now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
if args.text_first: 
    args.cols = (0,1)
    args.datafields = [("text", text_field), ("label", label_field)]
    args.names = ['text', 'label']
else: 
    args.cols = torch.tensor(1,0)
    args.datafields = [("label", label_field), ("text", text_field)]
    args.names = ['label', 'text']
#args.model = '{}_{}'.format(args.model, args.method)

# making path and save_dir Posixpath
args.path = Path(args.path)/args.dataset
args.save_dir = Path(args.save_dir)/args.now
if not args.save_dir.is_dir(): args.save_dir.mkdir()
# creating dated path for saving updated datasets later
if not (args.path/args.now).is_dir(): (args.path/args.now).mkdir()
# creating dataframes
print('\nCreatinf DataFrames ... \n')
train_df = pd.read_csv(args.path/'train.csv', header=None, names=args.names)
valid_df = pd.read_csv(args.path/'val.csv', header=None, names=args.names)
test_df = pd.read_csv(args.path/'test.csv', header=None, names=args.names)
# copying validation set to new dated path
print('Copying validation set to time specific folder. \n')
valid_df.to_csv(args.path/args.now/'val.csv', index=False, header=False)


# creating datasets 
train_ds = data.TabularDataset(path=args.path/'train.csv', format='csv', fields=args.datafields)
label_field.build_vocab(train_ds)
args.class_num = len(label_field.vocab) - 1
# creating DataBunch objects for langage modelling and classification
print('\nCreating DataBunch objects...')
data_lm = TextLMDataBunch.from_df(args.path, train_df=train_df, valid_df=valid_df, test_df=test_df, text_cols=0, label_cols=1)
data_clas = TextClasDataBunch.from_df(args.path, train_df=train_df, valid_df=valid_df, test_df=test_df, text_cols=0, label_cols=1,
                                      vocab=data_lm.train_ds.vocab, bs=args.bs)

# fine-tuning language model
print('\nFine-tuning language model ...')
helpers.language_model(data_lm, args)
# creating a classifier
print('\nTraining classifier ...')
model = helpers.classifier(data_clas, args)

for avg_iter in range(args.num_avg):
    if args.method is not None:
        al.al(model, avg_iter, args)
    else: 
        print('no method selected')
        break
