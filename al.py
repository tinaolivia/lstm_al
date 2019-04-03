#!/usr/bin/env python3
import pandas as pd
import methods
import helpers

import torchtext.data as data
from fastai.text import *

def al(init_model, avg_iter, args):
    
    train_df = pd.read_csv(args.path/'train.csv', header=None, names=args.names)
    test_df = pd.read_csv(args.path/'test.csv', header=None, names=args.names)
    
    print('\nEvaluating initial model ...')
    preds = init_model.validate()
    
    helpers.write_result(args.save_dir/'{}_{}_{}.csv'.format(args.dataset, args.method, avg_iter), 'w', ['Train Size', 'loss', 'accuracy', 'total {}'.format(args.method)], args)
    helpers.write_result(args.save_dir/'{}_{}_{}.csv'.format(args.dataset, args.method, avg_iter), 'a', [len(train_df['text']), preds[0], preds[1].numpy(), 0], args)
    
    model = init_model
    
    print('\nStarting active learning loop ...')
    for al_loop in range(args.rounds):
        
        # selecting new instances to add to train
        if args.method == 'random':
            subset = methods.random(test_df, args)
            total = 0
        
        if args.method == 'entropy':
            subset, total = methods.entropy(model, args)
            
        print('Round {}: {} insances selected according to {}'.format(al_loop,len(subset), args.method))
        
        # updating datasets
        print('\nUpdating datasets ...')
        #helpers.update_datasets(train, test, subset, args)
        helpers.update_datasets(train_df, test_df, subset, args)
        
        # reload data as DataBunch an retrain the model
        print('\nReloading data ...')
        train_df = pd.read_csv(args.path/args.now/'train_up.csv', header=None, names=args.names)
        valid_df = pd.read_csv(args.path/args.now/'val.csv', header=None, names=args.names)
        test_df = pd.read_csv(args.path/args.now/'test_up.csv', header=None, names=args.names)
        data_lm = TextLMDataBunch.from_df(args.path/args.now, train_df=train_df, valid_df=valid_df, test_df=test_df, text_cols=args.cols[0], label_cols=args.cols[1])
        data_clas = TextClasDataBunch.from_df(args.path/args.now, train_df=train_df, valid_df=valid_df, test_df=test_df, text_cols=args.cols[0], label_cols=args.cols[1], vocab=data_lm.train_ds.vocab, bs=args.bs)
        
        test_ds = data.TabularDataset(path=args.path/args.now/'test_up.csv', format='csv', fields=args.datafields)
        
        print('\nRetraining model ...')
        # fine tuning language model
        helpers.language_model(data_lm, args)
        # create a classifier
        model = helpers.classifier(data_clas, args)
        
        print('\nEvaluating ...')
        preds = model.validate()
        helpers.write_result(args.save_dir/'{}_{}_{}.csv'.format(args.dataset, args.method, avg_iter), 'a', [len(train_df), preds[0], preds[1].numpy(), total], args)

