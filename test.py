import os
import pdb
import time
import datetime

from data_structures import Corpus, Collection
from summarizers import multi_lead, summarize
from functions import learn_relevance, load

__author__ = 'matteo'


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

    print "\nTesting..."
    cp = load(read)
    (X, y, t) = cp.export_data()
    w = learn_relevance(X, y, reg_algo)

    sample_lead = multi_lead(cp.collections['d301i'+'2005'], sum_len)
    sample_regr = summarize(cp.collections['d301i'+'2005'], w, ext_algo, sum_len)

    print "\nPrint sample lead followed by sample regression:\n"
    for s in sample_lead:
        print s.strip()
    for s in sample_regr:
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
    print "summarize and store, test collections: %f seconds" % (time.time()-start)

    print "\nEvaluate on true feed..."
    c = Collection()
    c.read_test_collections("grexit")
    c.process_collection(False)
    start = time.time()
    summ = summarize(c, w, 'greedy',sum_len)
    for s in summ:
        print s.strip()
    print "signal feed: %f seconds" % (time.time() - start)

except Exception as e:

    print e
    pdb.set_trace()
