import numpy as np
from copy import deepcopy

def prune_weights(weight_list,pruning_rate):
    for i in range(weight_list.shape[-1]):
        copy_weight= deepcopy(weight_list[...,i])
        std=np.std(copy_weight)
        threshold= std*pruning_rate
        weight_list[...,i][np.abs(weight_list[...,i]) < threshold]=0
    return weight_list