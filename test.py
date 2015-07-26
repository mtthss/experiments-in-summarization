import os
import pdb
import time

from data_structures import Collection
from summarizers import multi_lead, summarize
from functions import learn_relevance, load, gen_name, plot_summary


__author__ = 'matteo'


print "\nConfiguring..."
reg_algo = "rf-R"
ext_algo = 'greedy'
read = False
human_inspect = False
store_test = False
sum_len = 6
d_name = gen_name(ext_algo, reg_algo)

print "\nLearn..."
cp = load(read)
(X, y, t) = cp.export_data()
w = learn_relevance(X, y, reg_algo)

print "\nGenerate..."
sample_lead = multi_lead(cp.collections['d301i'+'2005'], sum_len)
sample_regr = summarize(cp.collections['d301i'+'2005'], w, ext_algo, sum_len)

print "\nPrint sample lead followed by sample regression:"
plot_summary(sample_lead)
plot_summary(sample_regr)

if store_test:
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
summ = summarize(c, w, 'greedy',sum_len)
plot_summary(summ)

