#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pandas as pd
from fastai.text import *

import eval_w_dropout


def random(data_size, args):
    '''
        input:
        data_size: number of data samples 
    '''
    randperm = torch.randperm(data_size)
    subset = list(randperm[:args.inc].numpy())
    return subset


def entropy(model, args):
    '''
        input:
        model: current model
    '''
    preds = model.get_preds(DatasetType.Test)[0].cuda()
    entropy = (-preds*torch.log(preds)).sum(dim=1).cuda()
    sorted_e, sorted_ind = entropy.sort(0,True)
    total_e = sorted_e[:args.inc].sum()
    subset = list(sorted_ind[:args.inc].numpy())
    return subset, total_e 

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
    