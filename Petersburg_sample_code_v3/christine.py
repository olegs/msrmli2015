from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

__author__ = 'oleg'

def christine(D, n_estimators):
    clf = Pipeline([
        ('feature_selection', LinearSVC(penalty="l1", dual=False)),
        ('classification', RandomForestClassifier(n_estimators, random_state=1))
    ])
    return clf.fit(D.data['X_train'], D.data['Y_train'])
    # return RandomForestClassifier(n_estimators, random_state=1).fit(D.data['X_train'], D.data['Y_train'])
