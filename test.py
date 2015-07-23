__author__ = 'matteo'

from heapq import heappush, nlargest
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re
from scipy import spatial
import time
from nltk import FreqDist
from nltk.tag.mapping import map_tag
import kenlm
from nltk.util import ngrams
from nltk.stem import PorterStemmer
import datetime
import os


pos_tagger = nltk.pos_tag
LModel = model = kenlm.LanguageModel('kenlm/bible.klm') # http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html, NEURAL-LM https://github.com/pauldb89/OxLM
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = PorterStemmer()


print "\n\n\n\n"
text= 'tagged: approval, ex Communist states, EZ parliaments, Germany, government, Greece, Greek proposals, Grexit, Lithuania President, reactions, table Support "Greece"'
text2 = 'It turns out that locking Eurozone representatives in a room for 17 hours can produce some sort of resolution.'
text3 = 'what random has might verb treasure good razor not meaningful total distribution fucked up night sentence not yet book"'
LModel = model = kenlm.LanguageModel('kenlm/bible.klm')
print LModel.score(text)
print LModel.score(text2)
print LModel.score(text3)


print "\n\n\n\n"
t = datetime.datetime.now().month
d = datetime.datetime.now().day
h = datetime.datetime.now().hour
m = datetime.datetime.now().minute
d_name = "./results/"+str(t)+"-"+str(d)+"-"+str(h)+"-"+str(m)
os.mkdir(d_name)
