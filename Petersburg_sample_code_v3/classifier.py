import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline

__author__ = 'ML2015Pluto'


# Feature selection inspired by some of the top teams
def classify(d, name):
    if name == 'christine':
        return Pipeline([
            ('feature_selection', SelectPercentile(percentile=30, score_func=sklearn.feature_selection.f_classif)),
            ('classification', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1))
        ]).fit(d.data['X_train'], d.data['Y_train'])

    elif name == 'jasmine':
        return Pipeline([
            ('feature_selection', SelectPercentile(percentile=35, score_func=sklearn.feature_selection.f_classif)),
            ('classification', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1))
        ]).fit(d.data['X_train'], d.data['Y_train'])

    elif name == 'madeline':
        # See madeline.ipynb for more details
        return Pipeline([
            ('feature_selection', SelectPercentile(percentile=5, score_func=sklearn.feature_selection.f_classif)),
            ('classification', RandomForestClassifier(n_estimators=250, random_state=1, n_jobs=-1))
        ]).fit(d.data['X_train'], d.data['Y_train'])

    elif name == 'philippine':
        return Pipeline([
            ('feature_selection', SelectPercentile(percentile=35, score_func=sklearn.feature_selection.f_classif)),
            ('classification', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1))
        ]).fit(d.data['X_train'], d.data['Y_train'])

    elif name == 'sylvine':
        return Pipeline([
            ('feature_selection', SelectPercentile(percentile=50, score_func=sklearn.feature_selection.f_classif)),
            ('classification', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=-1))
        ]).fit(d.data['X_train'], d.data['Y_train'])

    else:
        raise AssertionError("FAILED to load classifier for " + name)
