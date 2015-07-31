from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

__author__ = 'oleg'

def philippine(D, n_estimators):
    clf = Pipeline([
        ('feature_selection', VarianceThreshold(.5)),
        ('classification', RandomForestClassifier(n_estimators, random_state=1))
    ])
    return clf.fit(D.data['X_train'], D.data['Y_train'])