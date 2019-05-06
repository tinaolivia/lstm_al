#!/usr/bin/env python3
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from fastai.text import *
from fastai.callbacks.tracker import *
from fastai.callbacks import CSVLogger, SaveModelCallback

def language_model(data_lm, args):
    '''
        input:
        data_lm: TextLMDataBunch object
    '''
    # defining language model
    learn = language_model_learner(data_lm, AWD_LSTM, pretrained=True, drop_mult=0.3,
                               callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.0001, patience=3)])
    # training frozen
    learn.fit_one_cycle(1, args.lr, moms=(0.8,0.7))
    learn.save('fit_head')
    # training unfrozen
    learn.load('fit_head')
    learn.unfreeze()
    learn.fit_one_cycle(10, args.lr/10, moms=(0.8, 0.7))
    learn.save_encoder(args.model)

    
def classifier(data_clas, args):
    '''
        input:
        data_clas: TextClasDataBunch object
    '''
    # defining classifier
    learn = text_classifier_learner(data_clas, AWD_LSTM, pretrained=True, drop_mult=0.5,
                                callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.0001, patience=3)])
    learn.load_encoder(args.model)
    # fitting freezed model
    learn.fit_one_cycle(1,2e-2, moms=(0.8, 0.7))
    learn.save('first_{}'.format(args.model))
    # unfreezing one layer
    learn.load('first_{}'.format(args.model))
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
    learn.save('second_{}'.format(args.model))
    # unfreezing one more layer
    learn.load('second_{}'.format(args.model))
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/33, 5e-3), moms=(0.8, 0.7))
    learn.save('third_{}'.format(args.model))
    # unfreezing model
    learn.load('third_{}'.format(args.model))
    learn.unfreeze()
    learn.fit_one_cycle(100, slice(1e-3/33, 1e-3), moms=(0.8, 0.7))
    return learn


def clustering(data, args):
    '''
        input:
        data: dataframe column containing text
    '''
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    kmeans = MiniBatchKMeans(n_clusters=args.inc).fit_predict(X)
    kmeans = torch.tensor(kmeans)
    if args.cuda: kmeans = kmeans.cuda()
    return kmeans


def update_datasets(train, test, subset, args):
    '''
        input:
        train: dataframe object of the train set
        test: dataframe object of the test set
        subset: set of indices that will be taken from test and added to train
    '''
    train = train.append(test.iloc[subset])
    test = test.drop(subset)
    train.to_csv(args.path/args.now/'train_up.csv', index=False, header=False)
    test.to_csv(args.path/args.now/'test_up.csv', index=False, header=False)
    
    
def write_result(filename, mode, result, args):
    '''
        input:
        filename: path and filename
        mode: writing mode, w=crate new file, a=append to existing
        result: list containing results to add to file
    '''
    with open(filename, mode=mode) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(result)

