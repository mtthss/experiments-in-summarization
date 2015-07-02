import os
import xml.etree.ElementTree as ET
import nltk
import time
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.util import ngrams
from multiprocessing.pool import Pool
from contextlib import closing


__author__ = 'matteo'


# utility function for multiprocessing map
def initialize_collection(params_bundle):
    c = Collection(params_bundle[0])
    c.readCollectionFromDir(params_bundle[1], params_bundle[2])
    c.process_collection()
    return c


# ensemble of collections to be used for training
class Corpus:

    def __init__(self, parallel_jobs):

        # initialize
        tok_path = 'tokenizers/punkt/english.pickle'
        col_path = './data/collections'
        sent_detector = nltk.data.load(tok_path)
        sent_detector.tokenize("  ".strip())

        # collect collections paths
        path_list = []
        for year in os.listdir(col_path):
            for code in os.listdir(col_path+"/"+year):
                if code!="duc2005_topics.sgml" and (code not in ["d408c", "d671g", "d442g"]):
                    path_list.append((sent_detector, year, code))

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
    def export_training_data(self):

        x_list = []
        y_list = []
        for c in self.collections.values():
            for d in c.docs.values():
                for s in d.sent.values():
                    x_list.append(s[1])
                    y_list.append(s[2])
        return (np.ndarray(x_list),np.ndarray(y_list))


# set of documents related to the same topic
class Collection:

    def __init__(self, tokenizer=None):

        self.sent_detector = tokenizer if tokenizer!=None else nltk.data.load('tokenizers/punkt/english.pickle')

        self.code = -1          # id of the collection
        self.topic_title = -1   # keywords / topic title
        self.topic_descr = -1   # description of expected content
        self.docs = {}          # documents to summarize: {id: document-object}
        self.references = {}    # human references: {id: reference-object}

    # read specified collection, including docs, topic and references
    def readCollectionFromDir(self, year, code):

        self.code = code

        # build paths
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
        for filename in os.listdir(doc_path):
            root = ET.parse(doc_path+"/"+filename).getroot()
            id = root.find('DOCNO').text
            self.docs[id] = Document(root.find('HEADLINE').text, root.find('TEXT').text, self)

        # read references
        for filename in os.listdir(ref_path):
            encod = filename.split(".")
            with open(ref_path+"/"+filename, 'r') as f:
                content = f.read()
            if encod[0].lower()==code[:-1]:
                self.references[encod[4]]=Reference(content,self)

    # process document, compute features, and if requested label data
    def process_collection(self, score=True):

        if score:
            for d in self.docs.values():
                d.process_score_document()
        else:
            for d in self.docs.values():
                d.process_document()


# document class, including processing methods
class Document:

    def __init__(self, headline, raw_text, collection=None):
        if collection==None: collection = Collection()
        self.headline = headline                        # headline of the document
        self.raw_text = raw_text.replace('\n', ' ')     # raw text of the document
        self.father = collection                        # reference to collection-object
        self.sent = {}                                  # {sentence-position: (raw_text, features, rel-score)

    def process_score_document(self):
        tokenized = self.father.sent_detector.tokenize(self.raw_text)
        count = 0
        for s in tokenized:
            self.sent[count] = (s, self.compute_features(s, count), self.compute_svr_score(s), self.compute_ranksvm_score(s))

    def process_document(self):
        tokenized = self.father.sent_detector.tokenize(self.raw_text)
        count = 0
        for s in tokenized:
            self.sent[count] = (s, self.compute_features(s))

    def compute_features(self, sentence, count):
        return (count)

    def compute_svr_score(self, sentence):  # see litRev file
        return max([ref.basic_sent_sim(sentence) for ref in self.father.references.values()])

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

    print "\ntesting tokenizer..."
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    print "\ntesting collection class..."

    c = Collection(sent_detector)
    c.readCollectionFromDir(2005,"d301i")
    print c.topic_title

    print "\ntesting document class..."

    d = Document("my_title", "my_doc", c)
    d.process_score_document()
    print d.father.topic_title

    print "\ntesting corpus class..."
    start_time = time.time()
    cp = Corpus(8) # optimal 6
    print "read and processed 47 collections in: "+str(time.time() - start_time)

    print "\ntesting exporting as matrix"
    (X,y) = cp.export_training_data()
    print X.shape, y.shape