from functools import partial
import numpy as np


__author__ = 'matteo'


# return partially applied function
def learn_relscore_function(X_rel, y, algorithm="svr"):

    # TODO python interfaces:
    # https://bitbucket.org/wcauchois/pysvmlight
    # https://pypi.python.org/pypi/svmlight

    if(algorithm=="svr"):
        # TODO http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        # TODO partially apply weights learned from svr
        pass
    elif(algorithm=="svm-rank"):
        # TODO http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
        # TODO partially apply weights learned from svm-rank
        pass
    elif(algorithm=="linear-reg"):
        # TODO simple implementation with scikit learn
        pass
    elif(algorithm=="lead"):
        weights = np.asarray([1, 0, 0])
    elif(algorithm=="test"):
        weights = np.asarray([0.55, 0.35, 0.005])
    else:
        raise Exception('Learn score function: Invalid algorithm')

    return weights


# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass


