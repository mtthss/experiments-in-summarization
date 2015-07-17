__author__ = 'matteo'

from heapq import heappush, nlargest
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re
from scipy import spatial

