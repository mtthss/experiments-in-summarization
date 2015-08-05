import os
import time

from data_structures import Collection
from gensim.models.word2vec import Word2Vec
from summarizers import multi_lead, rel_summarize, mmr_summarize
from functions import learn_relevance, load, gen_name, plot_summary


__author__ = 'matteo'


# evaluate given configuration
def evaluate_config(ext_algo, reg_algo, sum_algo, red_algo, tradeoff, word_len, max_sent):

    d_name = gen_name(ext_algo, reg_algo, red_algo, sum_algo)

    if red_algo=='w2v':
        print '\nLoading w2v model...'
        w2v_path = "../sentiment-mining-for-movie-reviews/Data/GoogleNews-vectors-negative300.bin"
        model = Word2Vec.load_word2vec_format(w2v_path, binary=True)  # C binary format

    print "\nLearn..."
    cp = load(read)
    (X, y, t) = cp.export_data()
    w = learn_relevance(X, y, reg_algo)

    print "\nInfo..."
    print "train shape: ", X.shape
    print "number of test collections: ", len(t)

    print "\nGenerate..."
    sample_lead_1 = multi_lead(cp.collections['d301i'+'2005'], word_len, max_sent)
    sample_lead_2 = multi_lead(cp.collections['D0601A'+'2006'], word_len, max_sent)
    sample_regr = rel_summarize(cp.collections['d301i'+'2005'], w, word_len, max_sent)
    sample_mmr = mmr_summarize(cp.collections['d301i'+'2005'], w, ext_algo, red_algo, word_len, max_sent, tradeoff)

    print "\nPrinting sample lead / regression..."
    plot_summary(sample_lead_1)
    plot_summary(sample_lead_2)
    plot_summary(sample_regr)
    plot_summary(sample_mmr)

    if store_test:

        print "\nGenerating summaries for test collections"
        os.mkdir(d_name)
        out_file = open(d_name+"/configurations","w")
        out_file.write("features: "+features+"\next_algo: "+ext_algo+"\nsum_algo: "+sum_algo)
        if sum_algo!="lead": out_file.write("\nregression_algo: "+reg_algo)
        if sum_algo=="mmr": out_file.write("\nred_algo: "+red_algo+"\ntradeoff: "+str(tradeoff))

        start = time.time()
        for c in t:

            if sum_algo == 'lead':
                summ = multi_lead(c, word_len, max_sent)
                out_file = open(d_name+"/"+c.code.lower()+"_"+sum_algo,"w")
            elif sum_algo == 'rel':
                summ = rel_summarize(c, w, word_len, max_sent)
                out_file = open(d_name+"/"+c.code.lower()+"_"+reg_algo+"-"+sum_algo,"w")
            elif sum_algo == 'mmr':
                summ = mmr_summarize(c, w, ext_algo, red_algo, word_len, max_sent, tradeoff)
                out_file = open(d_name+"/"+c.code.lower()+"_"+reg_algo+"-"+sum_algo+"-"+red_algo,"w")
            else:
                raise Exception('sum_algo: Invalid algorithm')

            if human_inspect:
                out_file.write("TOPIC\n")
                out_file.write(c.code+"; "+c.topic_title)
                out_file.write("\n\nDESCRIPTION\n")
                out_file.write(c.topic_descr)
                out_file.write("\n\nSUMMARY\n")
            for s in summ:
                out_file.write(s+"\n")

            out_file.close()

        print "summarize and store, test collections: %f seconds" % (time.time()-start)


# test system
if __name__ == '__main__':

    print "\nConfiguring..."

    features = "(P, F5, LEN, LM, VS1, VB, NN, CT, Q)"
    read = False
    human_inspect = False
    store_test = True

    ext_algo = 'greedy'
    reg_algo = 'decisionT'       # rf-R, linear-R, decisionT
    sum_algo = 'mmr'           # mmr, lead, rel
    red_algo = 'uniCosRed'      # simpleRed, uniCosRed

    tradeoff = 0.2
    word_len = 250
    max_sent = 200

    evaluate_config(ext_algo, reg_algo, sum_algo, red_algo, tradeoff, word_len, max_sent)

    exit()

    print "\nEvaluate on true feed..."
    c = Collection()
    c.read_test_collections("grexit")
    c.process_collection(False)
    summ = rel_summarize(c, w, word_len, max_sent)
    plot_summary(summ)