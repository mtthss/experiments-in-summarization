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
from nltk import FreqDist
from contextlib import closing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag.mapping import map_tag
from multiprocessing.pool import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


__author__ = 'matteo'


# grammars features: http://www.nltk.org/book/ch08.html
# dependencies trees: http://www.nltk.org/book/ch08.html
# coreferences https://github.com/dasmith/stanford-corenlp-python WITH EXAMPLE
# think handcrafted pronoun resolution rules (start with "this is" -> subst with last noun previous sent)

cachedStopWords = stopwords.words("english")
cachedStopPOStags = ['.', 'X', 'PRT', 'ADP', 'CONJ', 'ADV', 'DET']

pos_tagger = nltk.pos_tag
LModel = kenlm.LanguageModel('kenlm/bible.klm') # http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html, NEURAL-LM https://github.com/pauldb89/OxLM
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = PorterStemmer()


# utility function for multiprocessing map
def initialize_collection(params_bundle):
    print str(params_bundle[0]), str(params_bundle[1]), "start..."
    c = Collection()
    c.readCollectionFromDir(params_bundle[0], params_bundle[1])
    c.process_collection()
    print str(params_bundle[0]), str(params_bundle[1]), "done!"
    return c

def clean(txt, stop=False, stem=False):
    txt = re.sub('"|\'|-|\||\n|<|>|\\\\+', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    txt = txt.lower()
    txt = ''.join(i for i in txt if not i.isdigit())
    txt = ' '.join([stemmer.stem(word) for word in txt.split() if word not in cachedStopWords])
    return txt.strip()


# ensemble of collections to be used for training
class Corpus:

    def __init__(self, parallel_jobs, test_mode=False):

        # initialize
        col_path = './data/collections'

        # collect collections paths
        path_list = []
        for year in os.listdir(col_path):
            count = 0
            #if year=="2006": continue
            for code in os.listdir(col_path+"/"+year):
                if (code!="duc2005_topics.sgml") and (code!="duc2006_topics.sgml") and (code not in ["d408c", "d671g", "d442g"]):
                    path_list.append((year, code))
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
            self.collections[c.code+c.year] = c

        print "number of collections in dict: ", len(self.collections)

    # export data: X,y = train sentence and score. t = test collections
    def export_data(self):

        start = time.time()
        x_list = []
        y_list = []
        t = []
        count = 1

        for c in self.collections.values():
            if count%5 == 0:
                t.append(c)
            else:
                for d in c.docs.values():
                    for s in d.sent.values():
                        x_list.append(s[1])
                        y_list.append(s[2])
            count +=1

        X = np.asarray(x_list)
        y = np.asarray(y_list)

        print "exporting: %f seconds" % (time.time() - start)
        return (X,y,t)


# set of documents related to the same topic
class Collection:

    def __init__(self, tokenizer=None):

        self.code = -1          # id of the collection
        self.year = -1          # conference year

        self.topic_title = -1   # keywords / topic title
        self.topic_descr = -1   # description of expected content
        self.title_vsv = None   # title vector
        self.desc_vsv = None    # description vector
        self.docs = {}          # documents to summarize: {id: document-object}

        self.cv = None          # count-vectorizer on whole collection
        self.doc_BoW = None     # doc representation as bag of words
        self.tv = None          # count-vectorizer on whole collection
        self.doc_tfidf = None   # doc representation using tf-idf

        self.ref_BoW = None     # doc representation as bag of words
        self.ref_dict = {}      # set of references for this collection

    # read specified collection, including docs, topic and references
    def readCollectionFromDir(self, year, code):

        #initialize
        self.code = code
        self.year = year
        doc_path = "./data/collections/"+str(year)+"/"+code
        top_path = "./data/collections/"+str(year)+"/duc"+str(year)+"_topics.sgml"
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
            node = root.find('HEADLINE')

            if node!=None:
                id = root.find('DOCNO').text
                txt = root.find('TEXT').text
                hl = node.text
                texts.append(txt)
                hls.append(hl)
                self.docs[id] = Document(hl, txt, id, self)

        # read references
        for filename in os.listdir(ref_path):
            encod = filename.split(".")
            if encod[0].lower()==code[:-1].lower():
                with open(ref_path+"/"+filename, 'r') as f:
                    content = f.read()
                self.ref_dict[encod[4]] = content

        # process with count vectorizer
        try:
            self.cv = CountVectorizer(analyzer="word",stop_words=cachedStopWords,preprocessor=clean,max_features=5000,lowercase=True)
            self.doc_BoW = self.cv.fit_transform(texts+hls+[self.topic_title, self.topic_descr])
            self.ref_BoW = self.cv.transform([c for c in self.ref_dict.values()])
            self.title_vsv = self.cv.transform(self.topic_title)
            self.desc_vsv = self.cv.transform(self.topic_descr)
        except:
            pdb.set_trace()

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
        self.cv = CountVectorizer(analyzer="word",stop_words=cachedStopWords,preprocessor=clean,max_features=5000,lowercase=True)
        self.doc_BoW = self.cv.fit_transform(texts+hls)

    # process document, compute features, and if requested label data
    def process_collection(self, score=True):
        for d in self.docs.values():
            d.process_document(score)

    # basic labelling
    def basic_labelling(self, sent):
        l =[]
        for r_count in xrange(len(self.ref_dict)):
            sum = 0
            w_count = 0
            for word in nltk.tokenize.word_tokenize(clean(sent)):
                idx = self.cv.vocabulary_.get(word)
                if idx!=None:
                    sum += self.ref_BoW[r_count, idx]
                    w_count += 1
            if w_count != 0:
                l.append(float(sum)/w_count)
            else:
                print sent
                l.append(0)
            r_count += 1
        try:
            return max(l)
        except:
            pass

    # vector space modelling labelling
    def vsm_labelling(self, sent):

        val = 0
        sent_v = self.cv.transform([sent]).toarray()
        for ref in self.ref_dict.values():
            temp = 1 - spatial.distance.cosine(sent_v, self.cv.transform([ref]).toarray())
            if math.isnan(temp): temp = 0
            val = max(temp, val)
        return val

    # rougeN labelling
    def rougeN_labelling(self, sent):
        return 0

    # tf idf cosine similarity labelling
    def tfidf_labelling(self):
        return 0


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
        for s in sent_detector.tokenize(self.raw_text):
            # len(clean(s))<15
            if len(s)<50 or len(s)>350:  # modify simultaneously as line 21 of summarizers.py
                continue

            s1 = self.compute_score(s,"basic") if score else 0
            s2 = self.compute_score(s,"vsm")  if score else 0
            #s3 = self.compute_score(s,"n-rouge")  if score else 0
            self.sent[count] = (s, self.compute_features(s, count), s1)
            count += 1

    # compute sentence features
    def compute_features(self, s, count):

        tok_sent = nltk.tokenize.word_tokenize(s)
        stop_tok_sent = [x for x in tok_sent if x not in cachedStopWords]

        P = 1.0/count
        F5 = 1 if count <=5 else 0
        LEN = len(stop_tok_sent)/30.0
        LM = LModel.score(s)
        tag_fd = FreqDist(map_tag("en-ptb", "universal",tag) if map_tag("en-ptb", "universal",tag) not in cachedStopPOStags else "OTHER" for (word, tag) in pos_tagger(tok_sent))
        NN = tag_fd.freq("NOUN")
        VB = tag_fd.freq("VERB")

        CT = 1 - spatial.distance.cosine(self.hl_vsv_1.toarray(), self.father.cv.transform([s]).toarray())
        Q = 1 - spatial.distance.cosine(self.hl_vsv_1.toarray(), self.father.cv.transform([s]).toarray())
        VS1 = 1 - spatial.distance.cosine(self.hl_vsv_1.toarray(), self.father.cv.transform([s]).toarray())
        if math.isnan(VS1):
            VS1 = 0
            print self.father.code, self.id
        if math.isnan(CT):
            CT = 0
            print self.father.code, self.id
        if math.isnan(Q):
            Q = 0
            print self.father.code, self.id

        return (P, F5, LEN, LM, VS1, VB, NN, CT, Q)

    # score sentence wrt reference summaries (svr)
    def compute_score(self, sentence, method):
        if method == "basic":
            return self.father.basic_labelling(sentence)
        elif method == "n-rouge":
            return self.father.rougeN_labelling(sentence)
        elif method == "vsm":
            return self.father.vsm_labelling(sentence)
        else:
            raise Exception('Label training data: Invalid algorithm')


# main
if __name__ == '__main__':

    print "\ntesting corpus class..."
    start_time = time.time()
    cp = Corpus(16)
    print "read and processed "+str(len(cp.collections))+" collections in: "+str(time.time() - start_time)

    print "\ntesting exporting as matrix"
    (X,y,t) = cp.export_data()
    print "shape of X and y: ", X.shape, y.shape

    if False:
        pdb.set_trace()