import re
import time
import numpy as np
from heapq import heappush, nlargest, heappop


__author__ = 'matteo'


# summarize a collection
def multi_lead(collection, num_words, max_sent):

    start = time.time()
    h = []
    for d in collection.docs.values():
        for s in d.sent.values():
            lead_w = np.zeros(len(list(s[1])))
            lead_w[0] = 1
            rel = np.dot(np.asarray(list(s[1])), lead_w)
            if len(s[0])<350:       # modify simultaneously as line 270 of data_structures.py
                heappush(h, (-1*rel, s[0]))
    cw = cs = 0
    most_rel = []
    while cw<num_words and cs<max_sent:
        cand = heappop(h)
        add = len(cand[1].split())
        if (cw + add)<num_words:
            most_rel.append(re.sub('-',' ',re.sub('\s+', ' ', cand[1])).strip())
            cw += add
        cs += 1
    print "lead: %f seconds" % (time.time() - start)
    return most_rel

# summarize a collection according to relevance score
def rel_summarize(collection, clf, num_words, max_sent):

    start = time.time()
    h = []
    for d in collection.docs.values():
        for s in d.sent.values():
            rel = clf.predict(np.asarray(list(s[1])))
            if len(s[0])<350:       # modify simultaneously as line 270 of data_structures.py
                heappush(h, (-1*rel, s[0]))
    cw = cs = 0
    most_rel = []
    while cw<num_words and cs<max_sent:
        cand = heappop(h)
        add = len(cand[1].split())
        if (cw + add)<num_words:
            most_rel.append(re.sub('-',' ',re.sub('\s+', ' ', cand[1])).strip())
            cw += add
        cs += 1
    print "supervised, greedy extraction: %f seconds" % (time.time() - start)
    return most_rel

# maximum marginal relevance summarization
def mmr_summarize(collection, clf, algorithm, num_words, max_sent, tradeoff):

    if algorithm=="greedy":
        pass
    elif algorithm=='dyn-prog':
        pass
    elif algorithm=="A*":
        pass
    else:
        raise Exception('Extract Summary: Invalid algorithm')
    return None

# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass