#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from fastai.text import *

import eval_w_dropout
import helpers


def random(data_size, args):
    '''
        input:
        data_size: number of instances
    '''
    randperm = torch.randperm(data_size)
    subset = list(randperm[:args.inc])
    return subset

def random_w_clustering(data_size, df, args):
    '''
        input:
        data_size: number of instances
    '''
    kmeans = helpers.clustering(df, args)
    randperm = torch.randperm(data_size)
    subset = []
    for i in range(args.inc):
        for j in range(data_size):
            if kmeans[randperm[j]] == i: subset.append(randperm[j])
    return subset


def entropy(model, args, df=None):
    '''
        input:
        model: trained model
        df: dataframe column of text documents
    '''
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    preds = model.get_preds(DatasetType.Test)[0].cuda()
    entropy = (-preds*torch.log(preds)).sum(dim=1).cuda()
    sorted_e, sorted_ind = entropy.sort(0,True)
    if args.cluster and (df is not None):
        top_e = torch.empty(args.inc).cuda()
        top_ind = torch.empty(args.inc).cuda()
        for i in range(args.inc):
            for j in range(len(df)):
                if kmeans[sorted_ind[j]] == i:
                    top_e[i] = sorted_e[j]
                    top_ind[i] = sorted_ind[j]
    else:
        top_e = sorted_e[:args.inc]
        top_ind = sorted_ind[:args.inc]
                
    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_e.cpu().numpy().sum()

def margin(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    preds = model.get_preds(DatasetType.Test)[0].cuda()
    sorted_ = preds.sort(descending=True)[0].cuda()
    #print(len(sorted_), len(sorted_[0]), len(sorted_[1]))
    margins = sorted_[:,0] - sorted_[:,1]
    sorted_m, sorted_ind = margins.sort(0,descending=False)
    #print(len(margins))
    if args.cluster and (df is not None):
        top_m = torch.empty(args.inc).cuda()
        top_ind = torch.empty(args.inc).cuda()
        for i in range(args.inc):
            for j in range(len(df)):
                if kmeans[sorted_ind[j]] == i:
                    top_m[i] = sorted_m[j]
                    top_ind[i] = sorted_ind[j]
    else:
        top_m = sorted_m[:args.inc]
        top_ind = sorted_ind[:args.inc]
        
    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_m.cpu().numpy().sum()

def variation_ratio(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    preds = model.get_preds(DatasetType.Test)[0].cuda()
    var = 1 - preds.max(dim=1)[0].cuda()
    sorted_var, sorted_ind = var.sort(0, descending=True)
    if args.cluster and (df is not None):
        top_var = args.empty(args.inc).cuda()
        top_ind = args.empty(args.inc).cuda()
        for i in range(args.inc):
            for j in range(len(df)):
                if kmeans[sorted_ind[j]] == i:
                    top_var[i] = sorted_var[j]
                    top_ind[i] = sorted_ind[j]
    else:
        top_e = sorted_var[:args.inc].cuda()
        top_ind = sorted_var[:args.inc].cuda()
        
    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_e.cpu().numpy().sum()

#-------------------------------------------------------------------------------------------------------
# Dropout methods                

def dropout_variability(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    probs = []
    for i in range(args.num_preds):
        probs.append(eval_w_dropout.get_preds(model, DatasetType.Test)[0].cuda())
    probs = torch.stack(probs).cuda()
    mean = probs.mean(dim=0).cuda()
    var = torch.abs(probs - mean).sum(dim=0).sum(dim=1).cuda()
    # var = torch.pow(preds - mean, 2).sum(dim=0).sum(dim=1)
    sorted_var, sorted_ind = var.sort(descending=True)
    if args.cluster and (df is not None):
        top_var = torch.empty(args.inc)
        top_ind = torch.empty(args.inc)
        for i in range(args.inc):
            for j in range(len(sorted_var)):
                if kmeans[sorted_ind[j]] == i:
                    top_var[i] = sorted_var[j]
                    top_ind[i] = sorted_ind[j]
    else:
        top_var = sorted_var[:args.inc]
        top_ind = sorted_ind[:args.inc]
                
    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_var.cpu().numpy().sum()

def dropout_entropy(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    probs = []
    for i in range(args.num_preds):
        probs.append(eval_w_dropout.get_preds(model, DatasetType.Test)[0].cuda())
    probs = torch.stack(probs).cuda()
    mean = probs.mean(dim=0).cuda()
    entropies = -(mean*torch.log(mean)).sum(dim=1).cuda()
    sorted_e, sorted_ind = entropies.sort(descending=True)
    if args.cluster and (df is not None):
        top_e = torch.empty(args.inc)
        top_ind = torch.empty(args.inc)
        for i in range(args.inc):
            for j in range(len(sorted_e)):
                if kmeans[sorted_ind[j]] == i:
                    top_e[i] = sorted_e[j]
                    top_ind[i] = sorted_ind[j]
    else:
        top_e = sorted_e[:args.inc]
        top_ind = sorted_ind[:args.inc]

    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_e.cpu().numpy().sum()

def dropout_margin(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    probs = []
    for i in range(args.num_preds):
        probs.append(eval_w_dropout.get_preds(model, DatasetType.Test)[0].cuda())
    probs = torch.stack(probs).cuda()
    sorted_mean = probs.mean(dim=0).sort(descending=True)[0]
    margins = sorted_mean[:,0] - sorted_mean[:,1]
    sorted_m, sorted_ind = margins.sort(descending=False)
    if args.cluster and (df is not None):
        top_m = torch.empty(args.inc)
        top_ind = torch.empty(args.inc)
        for i in range(args.inc):
            for j in range(len(sorted_m)):
                if kmeans[sorted_ind[j]] == i:
                    top_m[i] = sorted_m[j]
                    top_ind[i] = sorted_m[j]
    else:
        top_m = sorted_m[:args.inc]
        top_ind = sorted_ind[:args.inc]

    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_m.cpu().numpy().sum()

def dropout_variation(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    probs = []
    for i in range(args.num_preds):
        probs.append(eval_w_dropout.get_preds(model, DatasetType.Test)[0].cuda())
    probs = torch.stack(probs).cuda()
    means = probs.mean(dim=0).cuda()
    var = 1 - means.max(dim=1).cuda()
    sorted_var, sorted_ind = var.sort(descending=True)
    if args.cluster and (df is not None):
        top_var = torch.empty(args.inc)
        top_ind = torch.empty(args.inc)
        for i in range(args.inc):
            for j in range(len(sorted_var)):
                if kmeans[sorted_ind[j]] == i:
                    top_var[i] = sorted_var[j]
                    top_ind[i] = sorted_ind[j]
    else:
        top_var = sorted_var[:args.inc]
        top_ind = sorted_ind[:args.inc]

    subset = list(top_ind.cpu().numpy())
    for i, idx in enumerate(subset): subset[i] = int(idx)
    return subset, top_var.cpu().numpy().sum()
