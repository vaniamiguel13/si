import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split


def cross_validate(model, dataset, scoring, cv: int = 3, test_size: float = 0.2) -> dict:
    scores = {
        'seeds': [],
        'train': [],
        'test': []
    }

    # para cada validação
    for i in range(cv):
        # get seed
        seed = np.random.randint(0, 1000)

        # guarda no dict
        scores['seeds'].append(seed)

        train, test = train_test_split(dataset, test_size, random_state=seed)

        # fazer fit do dataset de treino
        model.fit(train)

        if scoring is None:
            scores['train'].append(model.score(train))
            scores['test'].append(model.score(train))

        else:
            y_train = train.Y
            y_test = test.Y

            scores['train'].append(scoring(y_train, model.predict(train)))
            scores['test'].append(scoring(y_test, model.predict(train)))

        return scores
