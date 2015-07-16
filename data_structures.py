import os
import pdb
import nltk
import time
import math
import numpy as np
import xml.etree.ElementTree as ET

import re
import os
import kenlm

from scipy import spatial
from nltk.util import ngrams
from contextlib import closing
from nltk.corpus import stopwords
from collections import defaultdict
from multiprocessing.pool import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


__author__ = 'matteo'
cachedStopWords = stopwords.words("english")


# utility function for multiprocessing map
def initialize_collection(params_bundle):
    c = Collection(params_bundle[0], params_bundle[1])
    c.readCollectionFromDir(params_bundle[2], params_bundle[3])
    c.process_collection()
    return c

def clean(txt, stop=False, stem=False):
    txt = re.sub('"|\'|-|\||\n|<|>|\\\\+', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    txt = txt.lower()
    txt = ''.join(i for i in txt if not i.isdigit())

    txt = ''.join([word for word in txt.split() if word not in cachedStopWords])

    return txt.strip()

# ensemble of collections to be used for training
class Corpus:

    def __init__(self, parallel_jobs, test_mode=False):

        # initialize
        tok_path = 'tokenizers/punkt/english.pickle'
        col_path = './data/collections'
        LM_path = './kenlm-master/lm/test.arpa'

        # intialize models
        sent_detector = nltk.data.load(tok_path)
        # TODO to train bigger more accurate model https://kheafield.com/code/kenlm/estimation/
        # TODO https://github.com/kpu/kenlm
        # TODO http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
        self.model = kenlm.LanguageModel(LM_path)

        # collect collections paths
        count = 0
        path_list = []
        for year in os.listdir(col_path):
            for code in os.listdir(col_path+"/"+year):
                if code!="duc2005_topics.sgml" and (code not in ["d408c", "d671g", "d442g"]):
                    path_list.append((sent_detector, self.model, year, code))
                    if test_mode and count>10:
                        break
                    count += 1

        # read and process documents, use parallelism is possible
        if parallel_jobs>1:
            with closing(Pool(processes=parallel_jobs)) as pool:
                collection_list = pool.map(initialize_collection, path_list)
        else:
            collection_list = [initialize_collection(x) for x in path_list]

        # store result in a dictionary
        self.collections = {}
        for c in collection_list:
            self.collections[c.code] = c

    # read corpus from a specified directory
    def read(self, path):
        pass

    # from a pickled list of Collections
    def load(self, path):
        pass

    # export training data in matrix format
    def export_training_data_regression(self):

        x_list = []
        y_list = []

        for c in self.collections.values():
            for d in c.docs.values():
                for s in d.sent.values():
                    x_list.append(s[1])
                    y_list.append(s[2]+s[3]) #s[2]+s[3]

        X = np.asarray(x_list)
        y = np.asarray(y_list)
        return (X,y)


# set of documents related to the same topic
class Collection:

    def __init__(self, tokenizer=None, model=None):

        self.sent_detector = tokenizer if tokenizer!=None else nltk.data.load('tokenizers/punkt/english.pickle')
        self.model = model if model!=None else kenlm.LanguageModel('./kenlm-master/lm/test.arpa')
        self.cachedStopWords = stopwords.words("english")   # loaded stopwords list

        self.code = -1          # id of the collection
        self.topic_title = -1   # keywords / topic title
        self.topic_descr = -1   # description of expected content

        self.cv = None          # count vectorizer on whole collection
        self.doc_BoW = None     # doc representation as bag of words
        self.tv = None          # count vectorizer on whole collection
        self.doc_tfidf = None   # doc representation using tfidf

        self.docs = {}          # documents to summarize: {id: document-object}
        self.references = {}    # human references: {id: reference-object}

    # read specified collection, including docs, topic and references
    def readCollectionFromDir(self, year, code):

        #initialize
        self.code = code
        doc_path = "./data/collections/"+str(year)+"/"+code
        top_path = "./data/collections/"+str(year)+"/duc2005_topics.sgml"
        ref_path = "./data/references/"+str(year)

        # read topic
        tree_top = ET.parse(top_path)
        root = tree_top.getroot()
        for tp in root.findall("./topic"):
            if tp.find("./num").text.strip()==code:
                self.topic_title = tp.find("./title").text.strip()
                self.topic_descr = tp.find("./narr").text.strip()

        # read documents
        texts = []
        hls = []
        for filename in os.listdir(doc_path):
            root = ET.parse(doc_path+"/"+filename).getroot()
            id = root.find('DOCNO').text

            txt = root.find('TEXT').text
            if len(txt)<10:
                ps = root.findall('./TEXT//P')
                txt = "".join([par.text for par in ps])
            texts.append(txt)

            hl = root.find('HEADLINE').text
            if len(hl)<6:
                ps = root.findall('./HEADLINE//P')
                hl = "".join([par.text for par in ps])
            hls.append(hl)

            self.docs[id] = Document(hl, txt, id, self)

        # process with count vectorizer
        self.cv = CountVectorizer(analyzer="word",stop_words=self.cachedStopWords,preprocessor=clean,max_features=5000,lowercase=True)
        self.doc_BoW = self.cv.fit_transform(texts+hls)

        # read references
        for filename in os.listdir(ref_path):
            encod = filename.split(".")
            with open(ref_path+"/"+filename, 'r') as f:
                content = f.read()
            if encod[0].lower()==code[:-1]:
                self.references[encod[4]] = Reference(content,self)

    # read test collection for which you want to generate summaries
    def read_test_collections(self, feed):

        tst_path = "./data/feeds/"+feed
        texts = []
        hls = []

        for filename in os.listdir(tst_path):
            root = ET.parse(tst_path+"/"+filename).getroot()
            id = root.find('DOCNO').text
            hl = root.find('HEADLINE').text
            ct = root.find('TEXT').text
            self.docs[id] = Document(hl, ct, id, self)
            texts.append(ct)
            hls.append(hl)

        # process with count vectorizer
        self.cv = CountVectorizer(analyzer="word",stop_words=self.cachedStopWords,preprocessor=clean,max_features=5000,lowercase=True)
        self.doc_BoW = self.cv.fit_transform(texts+hls)


    # process document, compute features, and if requested label data
    def process_collection(self, score=True):
            for d in self.docs.values():
                d.process_document(score)


# document class, including processing methods
class Document:

    def __init__(self, headline, raw_text, my_id, collection=None):

        if collection==None: collection = Collection()      # ensure a collection object exists
        self.father = collection                            # reference to collection-object

        self.headline = headline                            # headline of the document
        self.raw_text = raw_text.replace('\n', ' ')         # raw text of the document
        self.id = my_id                                     # doc number

        self.hl_vsv_1 = None                                # vector space representation uni-grams
        self.sent = {}                                      # {sentence-position: (raw_text, features, rel-score)

    # compute features and if requested score sentences wrt references
    def process_document(self, score=True):

        # compute headline features
        self.hl_vsv_1 = self.father.cv.transform([self.headline])

        # compute sentence features
        count = 1
        for s in self.father.sent_detector.tokenize(self.raw_text):

            if len(clean(s))<15:
                continue

            s1 = self.compute_svr_score(s) if score else 0
            s2 = self.compute_ranksvm_score(s) if score else 0
            self.sent[count] = (s, self.compute_features(s, count), s1, s2)
            count += 1

    # compute sentence features (P, F5, LEN, LM, VS1)
    def compute_features(self, s, count):

        # TODO http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

        tok_sent = [x for x in nltk.tokenize.word_tokenize(s) if x not in self.father.cachedStopWords]

        P = 1.0/count
        F5 = 1 if count <=5 else 0
        LEN = 0 #len(tok_sent)
        LM = self.father.model.score(s)
        VS1 = 1 - spatial.distance.cosine(self.hl_vsv_1.toarray(), self.father.cv.transform([s]).toarray())

        if math.isnan(VS1):
            #print s, self.headline
            VS1 = 0

        return (P, F5, LEN, LM, VS1)

    # score sentence wrt reference summaries (svr)
    def compute_svr_score(self, sentence):  # see litRev file
        return max([ref.basic_sent_sim(sentence) for ref in self.father.references.values()])

    # score sentence wrt reference summaries (rank-svm)
    def compute_ranksvm_score(self, sentence):  # see litRev file
        num = float(sum([ref.rougeN_sent_sim(sentence) for ref in self.father.references.values()]))
        den = float(sum([r.tot_count_big for r in self.father.references.values()]))
        return num/den


# class for human reference, including methods for comparing
class Reference:

    def __init__(self, raw_text, collection):

        self.ref = raw_text.strip()              # raw text of the human summary
        self.unigram_dict = defaultdict(int)     # dictionary with n-gram counts
        self.bigram_dict = defaultdict(int)
        self.tot_count_uni = 0
        self.tot_count_big = 0

        self.cachedStopWords = stopwords.words("english")
        self.tokens = nltk.tokenize.word_tokenize(self.ref)

        for word in self.tokens:
            if word not in self.cachedStopWords:
                self.unigram_dict[word] += 1
                self.tot_count_uni += 1

        for big in ngrams(self.tokens,2):
            self.bigram_dict[big]+=1
            self.tot_count_big += 1

    def basic_sent_sim(self, sentence):    # numerator, see litRev for complete formula
        l =[]
        for word in nltk.tokenize.word_tokenize(sentence):
            if word not in self.cachedStopWords: l.append(self.unigram_dict[word])
        if len(l)==0:
            return 0
        return sum(l)/float(len(l))

    def rougeN_sent_sim(self, sentence):   # numerator, see litRev for complete formula
        l = 0
        tks = nltk.tokenize.word_tokenize(sentence)
        for big in ngrams(tks,2):
            l += self.bigram_dict[big]
        return l


# main
if __name__ == '__main__':

    print "\ntesting corpus class..."
    start_time = time.time()
    cp = Corpus(8) # optimal 6
    print "read and processed 50 collections (approx 1600 articles) in: "+str(time.time() - start_time)

    print "\ntesting exporting as matrix"
    (X,y) = cp.export_training_data_regression()
    print X.shape, y.shape

    if False:
        pdb.set_trace()