import re
import pdb
import time
import numpy as np
#import rnn_language

from functions import simple_red, uni_cos_red, rnn_group_red, cnn_red
from heapq import heappush, heapify, heappop, nsmallest


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
def rel_summarize(collection, clf, num_words, max_sent, idx):

    start = time.time()
    h = []
    for d in collection.docs.values():
        for s in d.sent.values():
            rel = clf.predict(s[1][idx])
            if len(s[0])<350:       # modify simultaneously as line 270 of data_structures.py
                heappush(h, (-1*rel, s[0]))
    cw = cs = 0
    most_rel = []
    while cw<num_words and cs<max_sent:
        try:
            cand = heappop(h)
        except:
            pdb.set_trace()
        add = len(cand[1].split())
        if (cw + add)<num_words:
            most_rel.append(re.sub('-',' ',re.sub('\s+', ' ', cand[1])).strip())
            cw += add
        cs += 1
    print "supervised, greedy extraction: %f seconds" % (time.time() - start)
    return most_rel

# maximum marginal relevance summarization
def mmr_summarize(collection, clf, ext_algo, red_algo, num_words, max_sent, tradeoff, idx, w2v=None):

    if ext_algo=="greedy":

        start = time.time()
        h = []
        dict = {}
        for d in collection.docs.values():
            for s in d.sent.values():
                rel = clf.predict(s[1][idx])
                if len(s[0])<350:       # modify simultaneously as line 270 of data_structures.py
                    heappush(h, (-1*tradeoff*rel, s[0]))
                    dict[s[0]]= rel
        h = nsmallest(max_sent, h)
        cs = 1
        cw = 0
        flag = False
        most_rel = []
        first = re.sub('-',' ',re.sub('\s+', ' ', heappop(h)[1])).strip()
        most_rel.append(first)
        cw += len(first.split())
        while cw<num_words-10 and cs<max_sent:
            if flag:
                evaluate_redundancy(h, dict, most_rel, tradeoff, red_algo, w2v)
                heapify(h)
                flag = False
            try:
                cand = heappop(h)
            except IndexError:
                break
            add = len(cand[1].split())
            if (cw + add)<num_words:
                most_rel.append(re.sub('-',' ',re.sub('\s+', ' ', cand[1])).strip())
                cw += add
                flag = True
            cs += 1
        print "supervised, greedy extraction: %f seconds" % (time.time() - start)
        return most_rel

    elif ext_algo=='dyn-prog':
        pass
    else:
        raise Exception('Extract Summary: Invalid algorithm')

# update overall score of each sentences according to the given redundancy measure
def evaluate_redundancy(c, d, summ, tradeoff, red_algo, w2v=None):

    if red_algo=="simpleRed":
        for i in xrange(len(c)):
            s = c[i][1]
            t = summ[-1]
            k = float(len(summ))
            rel = -1 * tradeoff * d[s]
            red = (1-tradeoff) * simple_red(s, t)
            prev_score = c[i][0]
            old_red_sum = (prev_score-rel)*(k-1)
            update = (old_red_sum + red)/k
            c[i] = (rel + update, s)
    elif red_algo=="uniCosRed":
        for i in xrange(len(c)):
            c[i] = (-1*tradeoff*d[c[i][1]] + (1-tradeoff)*uni_cos_red(c[i][1], summ), c[i][1])
    elif red_algo=="groupRnnEmbedding":
        for i in xrange(len(c)):
            c[i] = (-1*tradeoff*d[c[i][1]] + (1-tradeoff)*rnn_group_red(c[i][1], summ), c[i][1])
    elif red_algo=="cnn":
        for i in xrange(len(c)):
            s = c[i][1]
            t = summ[-1]
            k = float(len(summ))
            rel = -1 * tradeoff * d[s]
            red = (1-tradeoff) * cnn_red(s, t, w2v)
            prev_score = c[i][0]
            old_red_sum = (prev_score-rel)*(k-1)
            update = (old_red_sum + red)/k
            c[i] = (rel + update, s)
    else:
        raise Exception('Redundancy Measure: Invalid algorithm')

# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass
