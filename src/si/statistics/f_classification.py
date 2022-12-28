from scipy import stats


def f_classification(dataset):
    classes = dataset.get_classes()
    grupos = [dataset.X[dataset.Y == c] for c in classes]
    F, p = stats.f_oneway(*grupos)
    return F, p
