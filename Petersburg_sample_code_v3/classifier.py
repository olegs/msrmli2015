import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline

__author__ = 'ML2015Pluto'


def eliminate_features(x, y, min_features):
    """Tree-based feature selection: ExtraTreesClassifier"""
    clf = ExtraTreesClassifier()
    clf.fit(x, y).transform(x)
    alpha = 0.5
    features = [index for index, value in enumerate(clf.feature_importances_) if value > alpha]
    while len(features) < min_features:
        alpha /= 2
        features = [index for index, value in enumerate(clf.feature_importances_) if value > alpha]
    return features


def classify(d, name):
    """Feature selection inspired by some of the top teams"""
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
            ('classification',
             ExtraTreesClassifier(n_estimators=300, n_jobs=-1, max_depth=None, min_samples_split=1, random_state=1,
                                  max_features=10))
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
