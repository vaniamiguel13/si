from typing import List
import numpy as np
import itertools
from si.model_selection.cross_validate import cross_validate



def randomized_search_cv(model, dataset, parameter_dist, scoring, cv: int = 3, n_iter = 100, test_size: float = 0.2) -> dict:
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

