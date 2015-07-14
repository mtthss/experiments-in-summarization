from functools import partial
from sklearn import linear_model
from sklearn.svm import SVR
import numpy as np
import pdb


__author__ = 'matteo'


# return partially applied function
def learn_relscore_function(X_rel, y, algorithm="svr"):

    # TODO python interfaces:
    # https://bitbucket.org/wcauchois/pysvmlight
    # https://pypi.python.org/pypi/svmlight

    if(algorithm=="svr"):
        svr_lin = SVR(kernel='linear', C=100000000)
        svr_lin.fit(X_rel, y)
        weights = svr_lin.coef_
        print svr_lin.coef_
        print svr_lin.intercept_
    elif(algorithm=="svm-rank"):
        # TODO http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
        # TODO partially apply weights learned from svm-rank
        pass
    elif(algorithm=="linear-reg"):
        clf = linear_model.LinearRegression(fit_intercept=False)
        clf.fit (X_rel, y)
        weights = clf.coef_
    elif(algorithm=="lead"):
        weights = np.asarray([1, 0, 0, 0, 0]) # first intercept, then others
    elif(algorithm=="test"):
        weights = np.asarray([0.6, 0.35, 0.025, 0.025, 0]) # first intercept, then others
    else:
        raise Exception('Learn score function: Invalid algorithm')

    return weights


# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass