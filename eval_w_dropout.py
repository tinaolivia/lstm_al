#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fastai.callback import CallbackHandler
from fastai.core import ifnone, is_listy, camel2snake, noop
from fastai.torch_core import to_np, to_detach
import fastai.torch_core
from fastprogress.fastprogress import progress_bar

from torch import Tensor
from torch.utils.data import DataLoader
from fastprogress.fastprogress import MasterBar, ProgressBar
from typing import Optional, Iterator, Tuple, Union, List, Callable, NewType
from functools import partial
from enum import Enum

OptOptimizer = Optional[optim.Optimizer]
Rank0Tensor = NewType('OneEltTensor', Tensor)
LossFunction = Callable[[Tensor, Tensor], Rank0Tensor]
OptLossFunc = Optional[LossFunction]
PBar = Union[MasterBar, ProgressBar]
DatasetType = Enum('DatasetType', 'Train Valid Test Single Fix')


def get_preds(learner, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None,
              pbar:Optional[PBar]=None) -> List[Tensor]:
    "Return predictions and targets on `ds_type` dataset."
    lf = learner.loss_func if with_loss else None
    return get_preds_w_dropout(learner.model, learner.dl(ds_type), cb_handler=CallbackHandler(learner.callbacks),
                               activ=_loss_func2activ(learner.loss_func), loss_func=lf, n_batch=n_batch, pbar=pbar)
    
   
def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    out = model(*xb)
    out = cb_handler.on_loss_begin(out)

    if not loss_func: return to_detach(out), yb[0].detach()
    loss = loss_func(out, *yb)

    if opt is not None:
        loss,skip_bwd = cb_handler.on_backward_begin(loss)
        if not skip_bwd:                     loss.backward()
        if not cb_handler.on_backward_end(): opt.step()
        if not cb_handler.on_step_end():     opt.zero_grad()

    return loss.detach().cpu()


def get_preds_w_dropout(model:nn.Module, dl:DataLoader, pbar:Optional[PBar]=None, cb_handler:Optional[CallbackHandler]=None,
              activ:nn.Module=None, loss_func:OptLossFunc=None, n_batch:Optional[int]=None) -> List[Tensor]:
    "Tuple of predictions and targets, and optional losses (if `loss_func`) using `dl`, max batches `n_batch`."
    res = [torch.cat(o).cpu() for o in
           zip(*validate_w_dropout(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch))]
    if loss_func is not None: 
        with torch_core.NoneReduceOnCPU(loss_func) as lf: res.append(lf(res[0], res[1]))
    if activ is not None: res[0] = activ(res[0])
    return res


def validate_w_dropout(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             pbar:Optional[PBar]=None, average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.train()
    with torch.no_grad():
        val_losses,nums = [],[]
        if cb_handler: cb_handler.set_dl(dl)
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler)
            val_losses.append(val_loss)
            if not is_listy(yb): yb = [yb]
            nums.append(yb[0].shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses
       

loss_func_name2activ = {'cross_entropy_loss': F.softmax, 'nll_loss': torch.exp, 'poisson_nll_loss': torch.exp,
    'kl_div_loss': torch.exp, 'bce_with_logits_loss': torch.sigmoid, 'cross_entropy': F.softmax,
    'kl_div': torch.exp, 'binary_cross_entropy_with_logits': torch.sigmoid,
}

def _loss_func_name2activ(name:str, axis:int=-1):
    res = loss_func_name2activ[name]
    if res == F.softmax: res = partial(F.softmax, dim=axis)
    return res

def _loss_func2activ(loss_func):
    if getattr(loss_func,'keywords',None):
        if not loss_func.keywords.get('log_input', True): return
    axis = getattr(loss_func, 'axis', -1)
    # flattened loss
    loss_func = getattr(loss_func, 'func', loss_func)
    # could have a partial inside flattened loss! Duplicate on purpose.
    loss_func = getattr(loss_func, 'func', loss_func)
    cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name == 'mix_up_loss':
        loss_func = loss_func.crit
        cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name in loss_func_name2activ:
        if cls_name == 'poisson_nll_loss' and (not getattr(loss_func, 'log_input', True)): return
        return _loss_func_name2activ(cls_name, axis)
    if getattr(loss_func,'__name__','') in loss_func_name2activ:
        return _loss_func_name2activ(loss_func.__name__, axis)
    return noop
    
        
 
    
    
    
    
    