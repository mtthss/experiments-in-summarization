import time
import datetime
import numpy as np
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
options = {"linear-R" : LinearRegression(fit_intercept=False),                      # ! MEDIUM-GOOD
           "kernel-RR": KernelRidge(kernel='rbf', gamma=0.1),                       # MEMORY ERROR
           "bayes-RR" : BayesianRidge(fit_intercept=False, compute_score=True),     # EQ. to linear-R
           "rf-R": RandomForestRegressor(n_estimators=30),                          # ! VERY GOOD
           "gb-R": GradientBoostingRegressor(n_estimators=30),                      # ! MEDIUM-GOOD
           "gauss-PR": GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1),      # MEMORY ERROR
           "decision-T": DecisionTreeRegressor(max_depth=2)                         # ! VERY GOOD
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

def gen_name(ext_algo, reg_algo):
    mt = datetime.datetime.now().month
    d = datetime.datetime.now().day
    h = datetime.datetime.now().hour
    mn = datetime.datetime.now().minute
    id = str(mt)+"-"+str(d)+"-"+str(h)+"-"+str(mn)+"-"+ext_algo+"-"+reg_algo
    return "./results/"+id

def plot_summary(s):
    print ""
    for s in s:
        print s.strip()
