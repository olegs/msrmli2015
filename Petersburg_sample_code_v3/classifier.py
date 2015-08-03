from math import exp

import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, VarianceThreshold

from libscores import auc_cv
from libscores import bac_cv

__author__ = 'ML2015Pluto'

def rf_model(x, y, p, e):
    return Pipeline([
        ('variation_zero', VarianceThreshold(.1)),
        ('feature_selection', SelectPercentile(percentile=p, score_func=sklearn.feature_selection.f_classif)),
        ('classification', RandomForestClassifier(n_estimators=e, random_state=1, n_jobs=-1, min_samples_split=1))
    ]).fit(x, y), "SELECT+RF percentile=%d" % p + " n_estimators=%d" % e


def rf_no_var_model(x, y, p, e):
    return Pipeline([
        ('variation_zero', VarianceThreshold(exp(-10))),
        ('feature_selection', SelectPercentile(percentile=p, score_func=sklearn.feature_selection.f_classif)),
        ('classification', RandomForestClassifier(n_estimators=e, random_state=1, n_jobs=-1, min_samples_split=1))
    ]).fit(x, y), "SELECT+RF_NO_VAR percentile=%d" % p + " n_estimators=%d" % e


def et_model(x, y, p, e):
    return Pipeline([
        ('variation_zero', VarianceThreshold(.1)),
        ('feature_selection', SelectPercentile(percentile=p, score_func=sklearn.feature_selection.f_classif)),
        ('classification', ExtraTreesClassifier(n_estimators=e, n_jobs=-1, random_state=1, min_samples_split=1))
    ]).fit(x, y), "SELECT+ET percentile=%d" % p + " n_estimators=%d" % e


import time

# Since some tasks may stop beforehand, use slightly more
TIME_BUDGET = 5 * 60


def process(X, Y, model_function, metrics_function, best_model, best_metrics, best_label, best_p, start):
    # Check time!
    if time.time() - start > TIME_BUDGET:
        return best_model, best_metrics, best_label, best_p
    for e in [100, 200, 270, 300]:
        # Start optimization from previous point
        if best_p > 0:
            l, r = max(5, best_p - 25), min(80, best_p + 10)  # Since e is increasing, assymetric here
        else:
            l, r = 5, 80

        # Left
        model_l, label_l = model_function(X, Y, l, e)
        metrics_l = metrics_function(model_l, X, Y)
        if metrics_l > best_metrics:
            best_metrics = metrics_l;
            best_label = label_l;
            best_model = model_l
        print "Processed: %s" % label_l + " score: %f" % metrics_l

        # Rigth
        model_r, label_r = model_function(X, Y, r, e)
        metrics_r = metrics_function(model_r, X, Y)
        if metrics_r > best_metrics:
            best_metrics = metrics_r;
            best_label = label_r;
            best_model = model_r
        print "Processed: %s" % label_r + " score: %f" % metrics_r

        no_progress = 0
        while True:
            # Check time!
            if time.time() - start > TIME_BUDGET:
                return best_model, best_metrics, best_label, best_p
            # Median point
            p = (l + r) / 2
            model_p, label_p = model_function(X, Y, p, e)
            metrics_p = metrics_function(model_p, X, Y)
            if metrics_p > best_metrics:
                best_metrics = metrics_p;
                best_label = label_p;
                best_model = model_p;
                best_p = p;
                no_progress = 0
            else:
                no_progress += 1
            print "Processed: %s" % label_p + " score: %f" % metrics_p + ". No progress %d steps" % no_progress

            if metrics_l > metrics_r:
                r, model_r, metrics_r, label_r = p, model_p, metrics_p, label_p
            else:
                l, model_l, metrics_l, label_l = p, model_p, metrics_p, label_p
            if no_progress >= 3 or r - l <= 1:
                break

    return best_model, best_metrics, best_label, best_p


def optimize(name, X, Y):
    """Performs optimization for given dataset"""
    start = time.time()

    if name in ["christine", "jasmine", "madeline", "philippine", "sylvine"]:
        metrics_function = bac_cv
    else:
        metrics_function = auc_cv

    # Starting point
    model, metrics, label, p = None, 0, None, -1

    if X.shape[0] / X.shape[1] > 10 or X.shape[1] > 1000:
        model, metrics, label, p = process(X, Y, rf_no_var_model, metrics_function, model, metrics, label, p, start)
    else:
        model, metrics, label, p = process(X, Y, et_model, metrics_function, model, metrics, label, p, start)
    model, metrics, label, p = process(X, Y, rf_model, metrics_function, model, metrics, label, p, start)

    print "%s " % name + " best model: %s" % label + " metrics: %f" % metrics
    print "Time %dsec" % (time.time() - start)
    return model
