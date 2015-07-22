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


pos_tagger = nltk.pos_tag
LModel = model = kenlm.LanguageModel('kenlm/bible.klm') # http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html, NEURAL-LM https://github.com/pauldb89/OxLM
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = PorterStemmer()

def clean(txt, stop=False, stem=False):
    txt = re.sub('"|\'|-|\||\n|<|>|\\\\+', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    txt = txt.lower()
    txt = ''.join(i for i in txt if not i.isdigit())
    txt = ' '.join([stemmer.stem(word) for word in txt.split() if word not in cachedStopWords])
    return txt.strip()

cachedStopWords = stopwords.words("english")
cv = CountVectorizer(analyzer="word", stop_words=None, preprocessor=clean, max_features=5000, ngram_range=(1,2))

text = "Former panamanian leader General Manuel Antonio Noriega's leader defence against drug trafficking charges in Miami gained ground this week."

print clean(text)
a = cv.fit_transform([text])

test = "I don't know what is going on today cartel money leader defence leader"

b = cv.transform([test]).toarray()

print a
print b

w = ngrams(nltk.tokenize.word_tokenize("panamanian leader"),2)
print cv.get_feature_names()
for x in w:
    print " ".join(x)
    if cv.vocabulary_.get(" ".join(x))==None:
        print "fuck"
    else:
        print a[0,cv.vocabulary_.get(" ".join(x))]
        print b[0,cv.vocabulary_.get(" ".join(x))]



text= 'tagged: approval, ex Communist states, EZ parliaments, Germany, government, Greece, Greek proposals, Grexit, Lithuania President, reactions, table Support "Greece"'

text2 = 'It turns out that locking Eurozone representatives in a room for 17 hours can produce some sort of resolution.'

text3 = 'what random has might verb treasure good razor not meaningful total distribution fucked up night sentence not yet book"'

LModel = model = kenlm.LanguageModel('kenlm/bible.klm')

print LModel.score(text)
print LModel.score(text2)
print LModel.score(text3)

w = nltk.tokenize.word_tokenize("'don't: you think's it's rather amusing'")
print w

