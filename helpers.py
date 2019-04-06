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
    learn = language_model_learner(data_lm, AWD_LSTM, pretrained=args.pretrained, drop_mult=0.5)
    learn.callback_fns += [partial(CSVLogger, filename='logs')]
    # training frozen
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, args.lr)
    # training unfrozen
    learn.unfreeze()
    learn.fit_one_cycle(20, args.lr/10, callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy', name='args.model')])
    
    
    #learn = language_model_learner(data_lm, AWD_LSTM, pretrained=args.pretrained, drop_mult=0.5,
    #                               callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=args.earlystop, patience=3)])
    # finding a suitable learning rate
    #learn.lr_find()
    #learn.recorder.plot(return_fig=False, suggestion=True)
    #lr = learn.recorder.min_grad_lr
    #print('success')
    #learn.fit_one_cycle(1, args.lr)
    #learn.unfreeze()
    #learn.fit_one_cycle(1, args.lr/10)
    #learn.fit(args.epochs)
    #learn.unfreeze()
    #learn.fit(args.epochs)
    #learn.save_encoder(args.model)

    
def classifier(data_clas, args):
    '''
        input:
        data_clas: TextClasDataBunch object
    '''
    learn = text_classifier_learner(data_clas, AWD_LSTM, pretrained=args.pretrained, drop_mult=0.5)
    #                                callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=args.earlystop, patience=3)])
    learn.load_encoder(args.model)
    learn.fit_one_cycle(1, args.lr)
    #learn.unfreeze()
    #learn.fit_one_cycle(2, args.lr)
    #learn.fit(args.epochs)
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
    learn.unfreeze()
    learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
    #learn.fit(args.epochs)
    #learn.unfreeze()
    #learn.fit(args.epochs)
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

