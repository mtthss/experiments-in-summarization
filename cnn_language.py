import cPickle as pk
import numpy as np
import nltk
import chainer.functions as F
from chainer import Variable
from gensim.models.word2vec import Word2Vec
import pdb


__author__ = 'matteo'


# load trained model
print "initializing cnn..."
cnn_model,max_pool_window_1,max_pool_stride_1,avg_pool_window_2,avg_pool_stride_2,max_pool_window_2,max_pool_stride_2 \
    = pk.load(open("./pickles/binary_27,29,2,8,4,8","r"))

# execute
def forward(x_data, y_data=None):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.relu(cnn_model.l1(x))
    h1 = F.max_pooling_2d(h1,max_pool_window_1,stride=max_pool_stride_1)
    h2 = F.dropout(F.relu(cnn_model.l2(h1)))
    h2 = F.average_pooling_2d(h2, avg_pool_window_2, stride=avg_pool_stride_2)
    h2 = F.max_pooling_2d(h2,max_pool_window_2,stride=max_pool_stride_2)
    #y = cnn_model.l3(h2)
    #return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    return h2

# generate sentence matrix to be fed to the net
def convert(s, w2v_model):
    sent_matrix = np.zeros((1, 1, 300, 43))
    count = 0
    for token in nltk.tokenize.word_tokenize(s):
        if count<43:
            try:
                sent_matrix[:,:,:,count] = w2v_model[token]
            except KeyError:
                pass
            count += 1
        else:
            break
    return sent_matrix

# generate sentence embedding
def embed_sent(s, w2v_model):
    embedding = forward(convert(s, w2v_model))
    return embedding.data.reshape(embedding.data.shape[0]*embedding.data.shape[1]*embedding.data.shape[2]*embedding.data.shape[3])

# test embedding
def test():

    import time
    sent = "how is it going my friend, why are you so slow? how is it going my friend"
    sent2 = "how is it going my friend, welcome to a new world, horrible film really"
    print "loading w2v..."
    s = time.time()
    w2v_path = "../sentiment-mining-for-movie-reviews/Data/GoogleNews-vectors-negative300.bin"
    w2v_model = Word2Vec.load_word2vec_format(w2v_path, binary=True)  # C binary format
    print time.time()-s

    print "embedding..."
    s = time.time()
    d = embed_sent(sent,w2v_model)
    print d
    print time.time()-s

    pdb.set_trace()

    print "embedding2..."
    s = time.time()
    d = embed_sent(sent2,w2v_model)
    print d
    print time.time()-s

    print "embedding2..."
    s = time.time()
    d = embed_sent(sent2,w2v_model)
    print d
    print time.time()-s

    print "embedding2..."
    s = time.time()
    d = embed_sent(sent2,w2v_model)
    print d
    print time.time()-s
