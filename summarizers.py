from functions import learn_relscore_function
from data_structures import Corpus, Collection
from heapq import heappush, nlargest
import numpy as np
import pdb
import time


__author__ = 'matteo'


# summarize a collection
def summarize(collection, weights, algorithm, num_sent):

    h = []

    if algorithm=='greedy':
        for d in collection.docs.values():
            for s in d.sent.values():
                rel = np.dot(np.asarray(list(s[1])), weights)
                if len(s[0])<350:
                    heappush(h, (rel, s[0]))

        most_rel = nlargest(num_sent,h)
        most_rel_txt = [sent[1] for sent in most_rel]
        return most_rel_txt

    elif algorithm=='dyn-prog':
        pass
    elif algorithm=="A*":
        pass
    else:
        raise Exception('Extract Summary: Invalid algorithm')

    return None


# extract lead sentences
def lead(collection):
    pass


# main
if __name__ == '__main__':

    print "Processing corpus..."
    start = time.time()
    cp = Corpus(1, test_mode=True) #optimal 14
    load_time = time.time()-start

    print "Exporting..."
    start = time.time()
    (X, y) = cp.export_training_data_regression()
    export_time = time.time()-start

    print "\nTesting lead..."
    w = learn_relscore_function(X, y, "lead")
    start = time.time()
    summ = summarize(cp.collections['d301i'], w, 'greedy',6)
    lead_time = time.time()-start
    for s in summ:
        print s.strip()

    print "\nLinear regression..."
    w = learn_relscore_function(X, y, "linear-reg")
    start = time.time()
    summ = summarize(cp.collections['d301i'], w, 'greedy',6)
    linreg_time = time.time()-start
    for s in summ:
        print s.strip()

    print "\nLoading read test feeds..."
    c = Collection()
    c.read_test_collections("grexit")
    c.process_collection(False)

    print "\nEvaluate on true feed (lin reg)..."
    start = time.time()
    summ = summarize(c, w, 'greedy',6)
    for s in summ:
        print s.strip()
    linreg_grexit_time = time.time()-start

    print "\nWeights..."
    print w

    print "\nProcessing: %f seconds" % load_time
    print "Exporting: %f seconds" % export_time
    print "Lead: %f seconds" % lead_time
    print "Lin Reg (train collection): %f seconds" % linreg_time
    print "Lin Reg (test collection): %f seconds" % linreg_grexit_time