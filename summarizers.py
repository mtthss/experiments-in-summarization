from functions import learn_relscore_function
from data_structures import Corpus, Collection
from heapq import heappush, nlargest
import numpy as np
import pdb


__author__ = 'matteo'


# summarize a collection
def summarize(collection, weights, algorithm, num_sent):

    h = []

    if algorithm=='greedy':
        for d in collection.docs.values():
            for s in d.sent.values():
                rel = np.dot(np.asarray(list(s[1])), weights)
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

    print "testing corpus..."
    cp = Corpus(1, test_mode=True) # optimal 6
    (X, y) = cp.export_training_data_regression()

    print "\n\ntesting lead..."
    w = learn_relscore_function(X, y, "lead")
    summ = summarize(cp.collections['d301i'], w, 'greedy',4)
    for s in summ:
        print s

    print "\n\nlinear regression..."
    w = learn_relscore_function(X, y, "linear-reg")
    summ = summarize(cp.collections['d301i'], w, 'greedy',4)
    for s in summ:
        print s

    print "testing read test feeds..."
    c = Collection()
    c.read_test_collections("grexit")
    c.process_collection(False)

    print "evaluate on true feed: lin reg"
    w = learn_relscore_function(X, y, "linear-reg")
    summ = summarize(c, w, 'greedy',4)
    for s in summ:
        print s

    print "\n\nsupport vector regression..."

    w = learn_relscore_function(X, y, "svr")
    print w
    summ = summarize(cp.collections['d301i'], w, 'greedy',4)
    for s in summ:
        print s