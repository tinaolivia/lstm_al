#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pandas as pd
from fastai.text import *

import eval_w_dropout
import helpers


def random(data_size, args):
    '''
        input:
        data_size: number of data samples 
    '''
    randperm = torch.randperm(data_size)
    subset = list(randperm[:args.inc].numpy())
    return subset

def random_w_clustering(data_size, df, args):
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
        model: current model
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
    return subset, top_e.cpu().numpy().sum()

def margin(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    preds = model.get_preds(DatasetType.Test)[0].cuda()
    sorted_ = preds.sort(descending=True)[0].cuda()
    margins = sorted_[0] - sorted_[1]
    sorted_m, sorted_ind = margins.sort(0,descending=False)
    if args.cluster and (df is not None):
        top_m = torch.empty(args.inc).cuda()
        top_ind = torch.empty(args.inc).cuda()
        for i in range(args.inc):
            for j in range(len(df)):
                if kmeans[sorted_ind[j]] == i:
                    top_m[i] = sorted_m[j]
                    top_ind = sorted_ind[j]
    else:
        top_e = sorted_m[:args.inc]
        top_ind = sorted_ind[:args.inc]
        
    subset = list(top_ind.cpu().numpy())
    return subset, top_e.cpu().numpy().sum()

def variation_ratio(model, args, df=None):
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    preds = model.get_preds(DatasetType.Test)[0].cuda()
    var = 1 - preds.max(dim=1)[0].cuda()
    sorted_var, sorted_ind = var.sort(0, descending=True).cuda()
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
    return subset, top_e.cpu().numpy().sum()

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Dropout methods                


def dropout(data, model, args):
    '''
        input:
        data: ds_type
        model: current model
    '''
    preds = torch.empty((args.num_preds, len(data), args.class_num)).cuda()
    for i in range(args.num_preds):
        preds[i,:,:] = eval_w_dropout.get_preds(model, DatasetType.Test)[0]
        
    mean = preds.mean(dim=0)
    var = torch.pow(preds - mean, 2).sum(dim=0).sum(dim=1)
    print(var.size())
    
    sorted_var, sorted_ind = var.sort(descending=True) 
    total_var = sorted_var[:args.inc].sum()
    subset = list(sorted_ind[:args.inc].numpy())
    return subset, total_var
    