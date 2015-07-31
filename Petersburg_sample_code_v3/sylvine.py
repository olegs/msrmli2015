from sklearn.ensemble import RandomForestClassifier

__author__ = 'oleg'

def sylvine(D, n_estimators):
    return RandomForestClassifier(n_estimators, random_state=1).fit(D.data['X_train'], D.data['Y_train'])