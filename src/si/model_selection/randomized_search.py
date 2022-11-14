from typing import List
import numpy as np
import itertools
from si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model, dataset, parameter_dist, scoring, cv: int = 3, n_iter=100, test_size: float = 0.2) -> \
        list[dict]:
    for parameter in parameter_dist.keys():
        assert hasattr(model, parameter), f'Model {model} não tem o parâmetro {parameter}'

    scores = []

    for i in range(n_iter):
        parameters = {k: np.random.choice(v) for k, v in parameter_dist.items()}

        for param, val in parameters.items():
            setattr(model, param, val)

        score = cross_validate(model, dataset, scoring, cv, test_size)

        score['parameters'] = parameters

        scores.append(score)

    return scores
