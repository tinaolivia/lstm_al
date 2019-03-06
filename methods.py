#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pandas as pd

def random(data, args):
    '''
        input:
        data: ds_type
    '''
    randperm = torch.randperm(len(data))
    subset = randperm[:args.inc]
    return subset


def entropy(data, model, args):
    '''
        input:
        data: ds_type
        model: current model
    '''
    preds = model.get_preds(data)[0].cuda()
    entropy = (-preds*torch.log(preds)).sum(dim=1).cuda()
    top_e, ind = entropy.sort(0,True)
    if args.cuda: top_e, ind = top_e.cuda(), ind.cuda()
    subset = ind[:args.inc]
    return subset    