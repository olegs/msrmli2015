from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

__author__ = 'ML2015Pluto'


# All the scores are taken from https://www.codalab.org/competitions/5211#results at 22:00(MSK) 01.08.2015
def classify(D, name, n_estimators):
    if name == 'christine':
        # 0.528 vs 0.504
        return Pipeline([
            ('feature_selection', VarianceThreshold(.5)),
            ('classification', RandomForestClassifier(n_estimators, random_state=1))
        ]).fit(D.data['X_train'], D.data['Y_train'])

    elif name == 'jasmine':
        # 0.646 vs 0.536
        return Pipeline([
            ('feature_selection', VarianceThreshold(.5)),
            ('classification', RandomForestClassifier(n_estimators, random_state=1))
        ]).fit(D.data['X_train'], D.data['Y_train'])

    elif name == 'madeline':
        # TODO BEST improvement is possible here!
        # 0.828 vs 0.599
        return Pipeline([
            ('feature_selection', VarianceThreshold(.9)),
            ('classification', RandomForestClassifier(n_estimators, random_state=1))
        ]).fit(D.data['X_train'], D.data['Y_train'])

    elif name == 'philippine':
        # 0.674 vs 0.547
        return Pipeline([
            ('feature_selection', VarianceThreshold(.5)),
            ('classification', RandomForestClassifier(n_estimators, random_state=1))
        ]).fit(D.data['X_train'], D.data['Y_train'])

    elif name == 'sylvine':
        # 0.948 vs 0.868
        # Score on Sylvine is good enough even with this algo
        return RandomForestClassifier(n_estimators, random_state=1).fit(D.data['X_train'], D.data['Y_train'])
    else:
        raise AssertionError("FAILED to load classifier for " + name)
