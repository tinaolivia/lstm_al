#!/usr/bin/env python3
import pandas as pd
import torch

from fastai.text import *
from fastai.callbacks.tracker import *
from pathlib import Path

def language_model(data_lm, args):
    '''
        input:
        data_lm: TextLMDataBunch object
    '''
    learn = language_model_learner(data_lm, AWD_LSTM, callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=args.earlystop, patience=3)])
    learn.unfreeze()
    learn.fit(args.epochs)
    learn.save_encoder(args.model)

    
def classifier(data_clas, args):
    '''
        input:
        data_clas: TextClasDataBunch object
    '''
    learn = text_classifier_learner(data_clas, AWD_LSTM, callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=args.earlystop, patience=3)])
    learn.load_encoder(args.model)
    learn.unfreeze()
    learn.fit(args.epochs)
    return learn


def update_datasets(train, test, subset, args):
    '''
        input:
        train: dataframe object of the train set
        test: dataframe object of the test set
        subset: set of indices that will be taken from test and added to train
    '''
    train = train.append(test.iloc[subset])
    test = test.drop(subset)
    train.to_csv(args.path/'train_up.csv', index=False, header=False)
    test.to_csv(args.path/'test_up.csv', index=False, header=False)
