from typing import List
import numpy as np
import itertools
from si.model_selection.cross_validate import cross_validate
from collections.abc import Callable


def randomized_search_cv(model, dataset, parameter_dist, scoring: Callable = None, cv: int = 3, n_iter=100,
                         test_size: float = 0.2) -> list[dict]:
    for parameter in parameter_dist:
        if not hasattr(model, parameter):
            raise AttributeError(f'Model {model} não tem o parâmetro {parameter}')

    scores = []

    for _ in range(n_iter):
        parameters = {}
        for param in parameter_dist:
            value = np.random.choice(parameter_dist[param])
            setattr(model, param, value)
            parameters[param] = value

        score = cross_validate(model, dataset, scoring, cv, test_size)
        score["parameters"] = parameters

        scores.append(score)

    return scores
