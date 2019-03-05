#!/usr/bin/env python3
import pandas as pd
import torch
import methods
import helpers
import csv

import torchtext.data as data
from fastai.text import *
from pathlib import Path

def al(model, avg_iter, args):
    
    #train = pd.read_csv(args.path/'train.csv', header=None, names=args.names)
    #test = pd.read_csv(args.path/'test.csv', header=None, names=args.names)
    train_df = pd.read_csv(args.path/'train.csv', header=None, names=args.names)
    test_df = pd.read_csv(args.path/'test.csv', header=None, names=args.names)
    test_ds = data.TabularDataset(path=args.path/'test.csv', format='csv', fields=args.datafields)
    
    print('\nEvaluating initial model ...')
    preds = model.validate()
    
    with open(args.save_dir/'{}_{}_{}.csv'.format(args.dataset, args.method, avg_iter), mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Train Size', 'Loss', 'Accuracy'])
        csvwriter.writerow([len(train_df), preds[0], preds[1].numpy()])
    
    print('\nStarting active learning loop ...')
    for al_loop in range(args.rounds):
        
        if args.method == 'random':
            subset = methods.random(test, args)
        
        # selecting new instances according to an active learning query strategy
        if args.method == 'entropy':
            subset = methods.entropy(test_ds, model, args)
            
        print('Round {}: {} insances selected according to {}'.format(al_loop,len(subset), args.method))
        
        # updating datasets
        print('\nUpdating datasets ...')
        #helpers.update_datasets(train, test, subset, args)
        helpers.update_datasets(train_df, test_df, subset, args)
        
        # reload data as DataBunch an retrain the model
        print('\nReloading data ...')
        data_lm = TextLMDataBunch.from_csv(args.path, csv_name='train_up.csv', test='val.csv', 
                                   text_cols=args.cols[0], label_cols=args.cols[1])
        data_clas = TextClasDataBunch.from_csv(args.path, csv_name='train_up.csv', test='val.csv', 
                                       text_cols=args.cols[0], label_cols=args.cols[1], vocab=data_lm.train_ds.vocab, bs=args.bs)
        train_df = pd.read_csv(args.path/'train_up.csv', header=None, names=args.names)
        test_df = pd.read_csv(args.path/'test_up.csv', header=None, names=args.names)
        test_ds = data.TabularDataset(path=args.path/'test_up.csv', format='csv', fields=args.datafields)
        
        print('\nRetraining model ...')
        # fine tuning language model
        helpers.language_model(data_lm, args)
        # create a classifier
        model = helpers.classifier(data_clas, args)
        
        print('\nEvaluating ...')
        preds = model.validate()
        with open(args.save_dir/'{}_{}_{}.csv'.format(args.dataset, args.method, avg_iter), mode='a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([len(train_df), preds[0], preds[1].numpy()])
