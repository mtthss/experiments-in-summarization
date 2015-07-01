
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
        self.topic_title = -1       # keywords / topic title
        self.topic_descr = -1       # description of expected content
        self.raw_docs = {}          # documents to summarize: {id: document-object}
        self.raw_references = {}    # human references: {id: raw-text}

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
            self.raw_docs[id] = Document(root.find('HEADLINE').text, root.find('TEXT').text, self)

        # read references
        for filename in os.listdir(ref_path):
            encod = filename.split(".")
            with open(ref_path+"/"+filename, 'r') as f:
                content = f.read()
            if encod[0].lower()==code[:-1]:
                self.raw_references[encod[4]]=content


class Document:

    def __init__(self, headline, raw_text, collection):
        self.headline = headline    # headline of the document
        self.raw_text = raw_text    # raw text of the document
        self.father = collection    # reference to collection-object
        self.sent = {}              # {sentence-position: (raw_text, features, rel-score)

    def process_document(self):

        pass



if __name__ == '__main__':

    print "testing collection class..."

    c = Collection()
    c.readCollectionFromDir(2005,"d301i")

    print c.topic_descr
    print c.topic_title