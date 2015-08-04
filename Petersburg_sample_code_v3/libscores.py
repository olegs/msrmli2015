# Score library for NUMPY arrays
# ChaLearn AutoML challenge

# For regression: 
# solution and prediction are vectors of numerical values of the same dimension

# For classification:
# solution = array(p,n) of 0,1 truth values, samples in lines, classes in columns
# prediction = array(p,n) of numerical scores between 0 and 1 (analogous to probabilities)

# Isabelle Guyon and Arthur Pesah, ChaLearn, August-November 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

import numpy as np
import scipy as sp
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import _check_cv, _fit_and_predict, _check_is_partition
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import indexable


def binarize_predictions(array):
    """ Turn predictions into decisions {0,1} by selecting the class with largest
    score for multiclass problems and thresholding at 0.5 for other cases."""
    # add a very small random value as tie breaker (a bit bad because this changes the score every time)
    # so to make sure we get the same result every time, we seed it    
    # eps = 1e-15
    # np.random.seed(sum(array.shape))
    # array = array + eps*np.random.rand(array.shape[0],array.shape[1])
    bin_array = np.zeros(array.shape)
    bin_array[array >= 0.5] = 1
    return bin_array


def acc_stat(solution, prediction):
    """ Return accuracy statistics TN, FP, TP, FN
     Assumes that solution and prediction are binary 0/1 vectors."""
    # This uses floats so the results are floats
    TN = sum(np.multiply((1 - solution), (1 - prediction)))
    FN = sum(np.multiply(solution, (1 - prediction)))
    TP = sum(np.multiply(solution, prediction))
    FP = sum(np.multiply((1 - solution), prediction))
    # print "TN =",TN
    # print "FP =",FP
    # print "TP =",TP
    # print "FN =",FN
    return (TN, FP, TP, FN)


def mvmean(R, axis=0):
    """ Moving average to avoid rounding errors. A bit slow, but...
    Computes the mean along the given axis, except if this is a vector, in which case the mean is returned.
    Does NOT flatten."""
    if len(R.shape) == 0: return R
    average = lambda x: reduce(lambda i, j: (0, (j[0] / (j[0] + 1.)) * i[1] + (1. / (j[0] + 1)) * j[1]), enumerate(x))[
        1]
    R = np.array(R)
    if len(R.shape) == 1: return average(R)
    if axis == 1:
        return np.array(map(average, R))
    else:
        return np.array(map(average, R.transpose()))


def bac_metric(solution, prediction):
    """
     Here we used hacked BAC to apply to solution without any wrapping
    Compute the normalized balanced accuracy. The binarization and
    the normalization differ for the multi-label and multi-class case. """
    bin_prediction = binarize_predictions(prediction)
    [tn, fp, tp, fn] = acc_stat(solution, bin_prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum(eps, tp)
    pos_num = sp.maximum(eps, tp + fn)
    tpr = tp / pos_num  # true positive rate (sensitivity)
    tn = sp.maximum(eps, tn)
    neg_num = sp.maximum(eps, tn + fp)
    tnr = tn / neg_num  # true negative rate (specificity)
    bac = 0.5 * (tpr + tnr)
    base_bac = 0.5  # random predictions for binary case
    bac = mvmean(bac)  # average over all classes
    # Normalize: 0 for random, 1 for perfect
    score = (bac - base_bac) / sp.maximum(eps, (1 - base_bac))
    return score


def tiedrank(a):
    """ Return the ranks (with base 1) of a list resolving ties by averaging.
     This works for numpy arrays."""
    m = len(a)
    # Sort a in ascending order (sa=sorted vals, i=indices)
    i = a.argsort()
    sa = a[i]
    # Find unique values
    uval = np.unique(a)
    # Test whether there are ties
    R = np.arange(m, dtype=float) + 1  # Ranks with base 1
    if len(uval) != m:
        # Average the ranks for the ties
        oldval = sa[0]
        newval = sa[0]
        k0 = 0
        for k in range(1, m):
            newval = sa[k]
            if newval == oldval:
                # moving average
                R[k0:k + 1] = R[k - 1] * (k - k0) / (k - k0 + 1) + R[k] / (k - k0 + 1)
            else:
                k0 = k;
                oldval = newval
    # Invert the index
    S = np.empty(m)
    S[i] = R
    return S


def auc_metric(solution, prediction):
    """ Normarlized Area under ROC curve (AUC).
    Return Gini index = 2*AUC-1 for  binary classification problems.
    Should work for a vector of binary 0/1 (or -1/1)"solution" and any discriminant values
    for the predictions. If solution and prediction are not vectors, the AUC
    of the columns of the matrices are computed and averaged (with no weight).
    The same for all classification problems (in fact it treats well only the
    binary and multilabel classification problems)."""
    # auc = metrics.roc_auc_score(solution, prediction, average=None)
    # There is a bug in metrics.roc_auc_score: auc([1,0,0],[1e-10,0,0]) incorrect
    label_num = 1
    auc = np.empty(label_num)
    r_ = tiedrank(prediction)
    s_ = solution
    if sum(s_) == 0: print('WARNING: no positive class example in class {}'.format(1))
    npos = sum(s_ == 1)
    nneg = sum(s_ < 1)
    auc[0] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
    return 2 * mvmean(auc) - 1


def bac_cv(m, x, y):
    predicted = cross_val_predict(m, x, y, cv=5, n_jobs=-1)
    return bac_metric(y, predicted)


def auc_cv(m, x, y):
    predicted = cross_val_predict(m, x, y, cv=5, n_jobs=-1)
    return auc_metric(y, predicted)


def cross_val_predict(estimator, X, y=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs'):
    """Generate cross-validated estimates for each input data point

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    cv : cross-validation generator or int, optional, default: None
        A cross-validation generator to use. If int, determines
        the number of folds in StratifiedKFold if y is binary
        or multiclass and estimator is a classifier, or the number
        of folds in KFold otherwise. If None, it is equivalent to cv=3.
        This generator must include all elements in the test set exactly once.
        Otherwise, a ValueError is raised.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    preds : ndarray
        This is the result of calling 'predict'
    """
    X, y = indexable(X, y)

    cv = _check_cv(cv, X, y, classifier=is_classifier(estimator))
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    preds_blocks = parallel(delayed(_fit_and_predict)(clone(estimator), X, y,
                                                      train, test, verbose,
                                                      fit_params)
                            for train, test in cv)
    p = np.concatenate([p for p, _ in preds_blocks])
    locs = np.concatenate([loc for _, loc in preds_blocks])
    if not _check_is_partition(locs, X.shape[0]):
        raise ValueError('cross_val_predict only works for partitions')
    preds = p.copy()
    preds[locs] = p
    return preds
