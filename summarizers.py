import os
import re
import pdb
import time
import datetime
import numpy as np
import cPickle as pk

from functions import learn_relevance
from data_structures import Corpus, Collection
from heapq import heappush, nlargest


__author__ = 'matteo'


# summarize a collection
def summarize(collection, clf, algorithm, num_sent):

    start = time.time()
    h = []

    if algorithm=='greedy':
        for d in collection.docs.values():
            for s in d.sent.values():
                rel = clf.predict(np.asarray(list(s[1])))
                if len(s[0])<350:       # modify simultaneously as line 270 of data_structures.py
                    heappush(h, (rel, s[0]))

        most_rel = nlargest(num_sent,h)
        most_rel_txt = [re.sub('-','',re.sub('\s+', ' ', sent[1])).strip() for sent in most_rel]
        print "supervised, greedy extraction: %f seconds" % (time.time() - start)
        return most_rel_txt

    elif algorithm=='dyn-prog':
        pass
    elif algorithm=="A*":
        pass
    else:
        raise Exception('Extract Summary: Invalid algorithm')

    return None

def greedy():
    pass

def dyn_prog():
    pass

def A_star():
    pass


# summarize a collection
def multi_lead(collection, num_sent):

    start = time.time()
    h = []
    for d in collection.docs.values():
        for s in d.sent.values():
            lead_w = np.zeros(len(list(s[1])))
            lead_w[0] = 1
            rel = np.dot(np.asarray(list(s[1])), lead_w)
            if len(s[0])<350:       # modify simultaneously as line 270 of data_structures.py
                heappush(h, (rel, s[0]))

    most_rel = nlargest(num_sent,h)
    most_rel_txt = [re.sub('-','',re.sub('\s+', ' ', sent[1])).strip() for sent in most_rel]
    print "lead: %f seconds" % (time.time() - start)
    return most_rel_txt

# choose best order for a set of extracted sentences
def reorder(sent_list, algorithm):
    pass

# process cross sentence references
def preprocess_crossreferences(corpus):
    pass


# main
if __name__ == '__main__':

    try:

        print "\nConfiguring..."
        reg_algo = "gb-R"
        ext_algo = 'greedy'
        read = False
        human_inspect = False
        sum_len = 6
        mt = datetime.datetime.now().month
        d = datetime.datetime.now().day
        h = datetime.datetime.now().hour
        mn = datetime.datetime.now().minute
        id = str(mt)+"-"+str(d)+"-"+str(h)+"-"+str(mn)+"-"+ext_algo+"-"+reg_algo
        d_name = "./results/"+id

        print "\nProcessing corpus..."
        start = time.time()
        if read:
            cp = Corpus(17)
        else:
            cp = pk.load(open("./pickles/corpus.pkl", "rb" ))
        load_time = time.time() - start

        print "\nPickling the corpus"
        start = time.time()
        if read:
            pk.dump(cp, open("./pickles/corpus.pkl", "wb"))
        pickle_time = start-time.time()

        print "\nExporting..."
        start = time.time()
        (X, y, t) = cp.export_data()
        export_time = time.time() - start

        print "\nLead, evaluate on crime and drugs..."
        start = time.time()
        summ = multi_lead(cp.collections['d301i'+'2005'], sum_len)
        lead_time = time.time() - start
        for s in summ:
            print s.strip()

        print "\nRegression, evaluate on crime and drugs..."
        w = learn_relevance(X, y, reg_algo)
        start = time.time()
        summ = summarize(cp.collections['d301i'+'2005'], w, ext_algo, sum_len)
        linreg_time = time.time() - start
        for s in summ:
            print s.strip()

        print "\nGenerating summaries for test collections"
        os.mkdir(d_name)
        start = time.time()
        for c in t:
            summ = summarize(c, w, ext_algo, sum_len)
            out_file = open(d_name+"/"+c.code.lower()+"_"+reg_algo,"w")
            if human_inspect:
                out_file.write("TOPIC\n")
                out_file.write(c.topic_title)
                out_file.write("\n\nDESCRIPTION\n")
                out_file.write(c.topic_descr)
                out_file.write("\n\nSUMMARY\n")
            for s in summ:
                out_file.write(s+"\n")
            out_file.close()
        store_test_time = time.time()-start

        print "\nLoading Signal test feeds..."
        c = Collection()
        c.read_test_collections("grexit")
        c.process_collection(False)

        print "\nEvaluate on true feed..."
        start = time.time()
        summ = summarize(c, w, 'greedy',sum_len)
        for s in summ:
            print s.strip()
        grexit_time = time.time() - start

        print "\nWeights..."
        print w

        print "\nProcessing: %f seconds" % load_time
        print "Pickling: %f seconds" % pickle_time
        print "Exporting: %f seconds" % export_time
        print "Lead: %f seconds" % lead_time
        print "Sample train collection: %f seconds" % linreg_time
        print "Summarize and store, test collections: %f seconds" % store_test_time
        print "Signal Feed: %f seconds" % grexit_time

    except Exception as e:

        print e
        pdb.set_trace()
