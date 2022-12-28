from typing import List
from collections.abc import Callable
import numpy as np
import itertools
from si.model_selection.cross_validate import cross_validate


def grid_search_cv(model, dataset, parameter_grid, scoring: Callable= None, cv: int = 3, test_size: float = 0.2) \
        -> list[dict]:
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f'Model {model} não tem o parâmetro {parameter}')

    scores = []

    for combination in itertools.product(*parameter_grid.values()):

        parameters = {}

        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        score = cross_validate(model, dataset, scoring, cv, test_size)

        score['parameters'] = parameters

        scores.append(score)

    return scores
