import os
import xml.etree.ElementTree as ET
import nltk


__author__ = 'matteo'


class Corpus:

    def __init__(self, input):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sent_detector.tokenize("  ".strip())

        # pass tokenizer to string
        self.collection_list = []

    # read corpus from a specified directory
    def readCorpusFromDir(self, path):
        pass

    # from a pickled list of Collections
    def loadCorpus(self, path):
        pass


class Collection:

    def __init__(self, tokenizer=None):

        self.sent_detector = tokenizer if tokenizer!=None else nltk.data.load('tokenizers/punkt/english.pickle')

        self.topic_title = -1   # keywords / topic title
        self.topic_descr = -1   # description of expected content
        self.docs = {}          # documents to summarize: {id: document-object}
        self.references = {}    # human references: {id: reference-object}

    def readCollectionFromDir(self, year, code):

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

    def compute_svr_score(self, sentence):
        return max([ref.basic_sent_sim(sentence) for ref in self.father.references.values()])

    def compute_ranksvm_score(self, sentence):
        return 0


class Reference:

    def __init__(self, raw_text, collection):

        self.ref = raw_text.strip()
        self.ref_sent = collection.sent_detector.tokenize(self.ref)

    def basic_sent_sim(self, sentence):
        return 0


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