import time
import nltk
import datetime
import numpy as np
import cPickle as pk

from scipy import spatial
from nltk.corpus import stopwords
from data_structures import Corpus
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


__author__ = 'matteo'


#initialize objects
cachedStopWords = stopwords.words("english")
cv = CountVectorizer(analyzer="word",stop_words=cachedStopWords,preprocessor=None,lowercase=True)
c_options = {"linear-R" : LinearRegression(fit_intercept=False),                      # ! MEDIUM-GOOD
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
        clf = c_options[algorithm]
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

# gen directory name for storing purposes
def gen_name(ext_algo, reg_algo, red_algo):
    mt = datetime.datetime.now().month
    d = datetime.datetime.now().day
    h = datetime.datetime.now().hour
    mn = datetime.datetime.now().minute
    id = str(mt)+"-"+str(d)+"-"+str(h)+"-"+str(mn)+"-"+ext_algo+"-"+reg_algo+"-"+red_algo
    return "./results/"+id

# print summary to std output
def plot_summary(s):
    print ""
    for s in s:
        print s.strip()

# compute redundancy between sentences using simple word match
def simple_red(s1, s2):
    ls1 = nltk.tokenize.word_tokenize(s1)
    ls2 = nltk.tokenize.word_tokenize(s2)
    return len([val for val in ls1 if val in ls2])/((len(ls1)+len(ls2))/2.0)

# unigram cosine similarity
def uni_cos_red(s, summ):
    vectors = cv.fit_transform([" ".join(summ), s])
    red = 1 - spatial.distance.cosine(vectors[0].toarray(), vectors[1].toarray())
    return red

# compute redundancy between sentences using neural embeddings
def distrib_red(s1, s2, method, model=None):
    ls1 = nltk.tokenize.word_tokenize(s1)
    ls2 = nltk.tokenize.word_tokenize(s2)
    if method == "w2v":
        v1 = np.zeros((1,300))
        v2 = np.zeros((1,300))
        for w in ls1:
            try: v1 += model[w]
            except: pass
        for w in ls2:
            try: v2 += model[w]
            except: pass
        try:
            red = 1 - spatial.distance.cosine(v1, v2)
        except Exception as e:
            print e
            red = 0
        return red

    elif method == "cnn":
        print "not yet available"
    elif method == "lstm":
        print "not yet available"
    else:
        raise Exception('Redundancy: Invalid algorithm')