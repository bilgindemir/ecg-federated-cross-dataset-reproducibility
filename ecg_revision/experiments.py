"""Explicit experiment plans derived from manuscript and saved run configurations."""
from dataclasses import replace
from itertools import product
from .common import Config

METHODS = ('fedavg', 'fedavgm', 'fedprox', 'fedadam')

def jobs(spec):
    base = Config(**spec['base'])
    result = []
    for method, seed, epochs, weighting in product(spec['methods'], spec['seeds'], spec.get('local_epochs',[1]), spec.get('weightings',['equal'])):
        settings = spec.get('method_settings',{}).get(method,{})
        cfg = replace(base, method=method, seed=seed, local_epochs=epochs, weighting=weighting, **settings)
        grid = spec.get('grids',{}).get(method)
        if grid:
            for values in product(*grid.values()):
                result.append(replace(cfg, **dict(zip(grid,values))).validate())
        else:
            result.append(cfg.validate())
    return result
