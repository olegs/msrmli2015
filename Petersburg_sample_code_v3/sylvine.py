from sklearn.ensemble import RandomForestClassifier

__author__ = 'oleg'

def sylvine(D, n_estimators):
    # Score on Sylvine is good enough even with this algo
    return RandomForestClassifier(n_estimators, random_state=1).fit(D.data['X_train'], D.data['Y_train'])