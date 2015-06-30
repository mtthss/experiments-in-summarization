
import os
import xml.etree.ElementTree as ET

__author__ = 'matteo'


class Corpus:

    def __init__(self, input):
        self.collection_list = []

    # read corpus from a specified directory
    def readCorpusFromDir(self, path):
        pass

    # from a pickled list of Collections
    def loadCorpus(self, path):
        pass

    # compute sentence features for redundancy and relevance
    def processCorpusRelevance(self, feat_rel):
        pass

    # label feature vectors according to reference summaries
    def generateTrainingRelevance(self, method):
        pass


class Collection:

    def __init__(self):
        self.topic_title = -1   # keywords / topic title
        self.topic_descr = -1   # description of expected content
        self.docs = {}          # documents to summarize: {id: (headline, text)}
        self.references = {}    # human references: {id: text}

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

            tree_docs = ET.parse(doc_path+"/"+filename)
            root = tree_docs.getroot()
            id = root.find('DOCNO').text
            hl = root.find('HEADLINE').text
            txt = root.find('TEXT').text
            self.docs[id] = (hl,txt)

        # read references
        for filename in os.listdir(ref_path):

            print filename

    def loadCollection(self, path):
        pass


class Document:

    def __init__(self, input):
        self.title = -1
        self.raw_sent = {}
        self.feat_rel = {}
        self.fear_red = {}


class Sentence:

    def __init__(self, input):
        self.tok_list = []


class Token:

    def __init__(self, input):
        self.word = ""
        self.Ner = False
        self.Centrality = -1


if __name__ == '__main__':

    print "testing collection class..."

    c = Collection()
    c.readCollectionFromDir(2005,"d301i")

    print c.topic_descr
    print c.topic_title