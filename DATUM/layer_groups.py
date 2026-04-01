import torch
import torch.nn as nn
import torch.nn.functional as F

def get_parameters(m): #get #nn.Parameters instances
    return [getattr(m, attr) for attr in dir(m) if isinstance(getattr(m, attr),nn.Parameter)]

def flatten_model(m):
    return [m] if not hasattr(m,'children') or len(list(m.children())) == 0 else \
        sum(map(flatten_model, list(m.children())), []) + get_parameters(m)

def norm_instance(l):
    return isinstance(l, nn.BatchNorm3d) or isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d) or \
           isinstance(l, nn.GroupNorm) or isinstance(l, nn.LayerNorm) or \
           isinstance(l, nn.InstanceNorm3d) or isinstance(l, nn.InstanceNorm2d) or isinstance(l, nn.InstanceNorm1d)

def get_groups(layer_groups, lrs=1e-3, wds=1e-3):
    opt_params = []
    if not isinstance(wds,list): wds = [wds]*len(layer_groups)
    if not isinstance(lrs,list): lrs = [lrs]*len(layer_groups)
    for i,g in enumerate(layer_groups):
        p_wd0 = [p for l in g if norm_instance(l) and hasattr(l,'named_parameters') \
                   for n,p in l.named_parameters() if p.requires_grad] + \
                [p for l in g if not norm_instance(l) and hasattr(l,'named_parameters') \
                   for n,p in l.named_parameters() if any(nd in n for nd in ['bias']) if p.requires_grad] + \
                [l for l in g if not hasattr(l,'named_parameters')] #nn.Parameters case
        p_wd =  [p for l in g if not norm_instance(l) and hasattr(l,'named_parameters') \
                   for n,p in l.named_parameters() if not any(nd in n for nd in ['bias']) if p.requires_grad]

        opt_params.append({'params': p_wd, 'weight_decay': wds[i],'lr':lrs[i]})
        opt_params.append({'params': p_wd0, 'weight_decay': 0.0,'lr':lrs[i]})
    return opt_params
