############################################################
# Recurrent Neural Network for the Ground State WF of the AIM
# Author: Jonas Rigo j.rigo@fz-juelich.de
# Date: 25/10/2023
############################################################

############################################################
# jax & flax & optax
############################################################
import flax.linen as nn
import flax
import jax.numpy as jnp

############################################################
# Python
############################################################
from typing import Any, Callable, Sequence
from functools import partial

############################################################
# stuff
############################################################
import numpy as np
import h5py

# back tracking resolution of the structure
def h5_iterate_nested_dict_save(g,d,verbosity = 0):
    for key, value in d.items():
        if isinstance(value, dict):
            if verbosity > 0: print(f"group {key}")
            if key in g:
                _g = g[key]
            else:
                _g = g.create_group(key)
            yield from h5_iterate_nested_dict_save(_g,value,verbosity)
        else:
            yield key, value, g

# back tracking resolution of the structure
def h5_iterate_nested_dict_load(g,d,verbosity = 0):
    for key, value in d.items():
        if isinstance(value, h5py.Group):
            if verbosity > 0: print(f"group {key}")
            g[key] = {}
            yield from h5_iterate_nested_dict_load(g[key],value,verbosity)
        else:
            yield key, value, g

class h5SaveParams(object):
    def __init__(self,file_name,mode='w'):
        self.file_name = file_name
        self.mode = mode
        f = h5py.File(self.file_name,mode)
        f.flush()
        f.close()

    def save_model_params(self, params, group_name, att, verbosity = 0):
        f = self.open()
        if group_name in f:
            group = f[group_name]
        else:
            group = f.create_group(group_name)
        for key, value in att.items(): group.attrs[key] = value
        for (key, value, g) in h5_iterate_nested_dict_save(group,flax.serialization.to_state_dict(params),verbosity):
            if verbosity > 0: print(f'dataset: {key}: {value}')
            if key in g: del g[key]
            g.create_dataset(key, data=value)
        self.close(f)

    def load_model_params(self, void_params, group_name, verbosity = 0):
        f = self.open()
        group = f[group_name]
        params = {}
        for (key, value, g) in h5_iterate_nested_dict_load(params,group,verbosity):
            if verbosity > 0: print(f'dataset: {key}: {value}')
            g[key] = value[()]  # Use [()] to get the value as a NumPy array
        self.close(f)
        return flax.serialization.from_state_dict(void_params,params)
    
    def close(self,f):
        f.flush()
        f.close()

    def open(self):
        f = h5py.File(self.file_name, 'a')
        return f