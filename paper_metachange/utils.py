import os
import sys
import time

import itertools
from itertools import izip_longest
import numpy as np

import cPickle as pickle

def make_dir(dirpath):
    if os.path.isfile(dirpath):
        print "Error: %s is a file!" %(dirpath)
        sys.exit(1)
    # Try to make the directory
    try:
        os.makedirs(dirpath)
    except OSError:
        pass
    

def grouper_nofill(iterable, n):
    it = iter(iterable)
    while True:
       chunk = list(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk


def timestamp():
    return time.strftime("%m-%d-%Y_%I:%M:%S")


def save_model_helper(fname, model, 
                      extra={}):
    """
    Save a fitted model, plus other values, as a pickle file.

    Args:
    - fname: filename to write model to
    - model: the fitted model (after sampling)

    Kwargs:
    - extra_dict: dictionary to include in the file (e.g.
      relevant parameters dictionary)
    """
    model_data = {"model": model}
    model_data.update(extra)
    with open(fname, "wb") as file_out:
        pickle.dump(model_data, file_out)


def save_as_pickle(fname, data, extra):
    to_save = {"data": data}
    to_save.update(extra)
    with open(fname, "wb") as file_out:
        pickle.dump(to_save, file_out)

        
def load_pickle(fname):
    data = {}
    with open(fname, "r") as file_in:
        data = pickle.load(file_in)
    return data


def replace(l, old_elt, new_elt):
    return [new_elt if e == old_elt else e for e in l]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def all_match(d, other_d):
    """
    Return True if every key, value pair
    in d matches other_d. other_d can be a
    super set of d.
    """
    val = True
    for k in d:
        if k not in other_d:
            val = False
            break
        if other_d[k] != d[k]:
            val = False
            break
    return val
                
