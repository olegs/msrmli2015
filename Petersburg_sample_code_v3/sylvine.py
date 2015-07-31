from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB

__author__ = 'oleg'


def sylvine(D, n_estimators):
    return BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=n_estimators / 10).fit(
        D.data['X_train'], D.data['Y_train'])