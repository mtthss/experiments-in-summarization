import time
import cPickle as pk

from data_structures import Corpus
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


__author__ = 'matteo'


# available relevance regressors
options = {"linear-R" : LinearRegression(fit_intercept=False),
           "kernel-RR": KernelRidge(kernel='rbf', gamma=0.1),
           "bayes-RR" : BayesianRidge(fit_intercept=False, compute_score=True),
           "rf-R": RandomForestRegressor(n_estimators=30),
           "gb-R": GradientBoostingRegressor(n_estimators=30),
           "gauss-PR": GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1),
           "decision-T": DecisionTreeRegressor(max_depth=2),
           "lead" : None,
           }

# return partially applied function
def learn_relevance(X_rel, y, algorithm="linear-R"):
    start = time.time()
    try:
        clf = options[algorithm]
    except:
        raise Exception('Learn score function: Invalid algorithm')

    clf.fit (X_rel, y)
    print "regression: %f seconds" % (time.time() - start)
    print "training error: ", mean_squared_error(y, clf.predict(X_rel))
    return clf

# load corpus
def load(read):
    start = time.time()
    if read:
        cp = Corpus(17)
    else:
        cp = pk.load(open("./pickles/corpus.pkl", "rb" ))
    print "processing: %f seconds" % (time.time() - start)
    start = time.time()
    if read:
        pk.dump(cp, open("./pickles/corpus.pkl", "wb"))
        print "pickling: %f seconds" % (start - time.time())
    return cp



