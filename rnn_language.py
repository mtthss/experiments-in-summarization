import math
import cPickle as pk
import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F


__author__ = 'matteo'


print "initializing rnn..."
batchsize = 1
units = pk.load(open("./pickles/unitssmall.pkl","r"))
vocab = pk.load(open("./pickles/vocab.pkl","r"))
model = pk.load(open("./pickles/modelsmall.pkl","r"))


def convert(txt):
    words = txt.replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            dataset[i] = vocab["<UKN>"]
        else:
            dataset[i] = vocab[word]
    return dataset


def forward_one_step(x_data, y_data, state):
    x = chainer.Variable(x_data, volatile=True)
    t = chainer.Variable(y_data, volatile=True)

    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=False)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=False)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(h2)

    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, F.softmax_cross_entropy(y, t)


def forward_one_step_embed(x_data, state):

    x = chainer.Variable(x_data, volatile=True)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=False)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=False)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    return {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}


def make_initial_state(batchsize=batchsize, train=False):
    return {name: chainer.Variable(np.zeros((batchsize, units),dtype=np.float32),volatile=not train) for name in ('c1', 'h1', 'c2', 'h2')}


def evaluate_ppx(dataset):
    sum_log_perp = np.zeros(())
    state = make_initial_state(batchsize=batchsize, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = dataset[i:i + 1]
        y_batch = dataset[i + 1:i + 2]
        state, loss = forward_one_step(x_batch, y_batch, state)
        sum_log_perp += loss.data.reshape(())
    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))


def embed_sentence(dataset):
    state = make_initial_state(batchsize=batchsize, train=False)
    for i in six.moves.range(dataset.size - 1):
        x_batch = dataset[i:i + 1]
        state = forward_one_step_embed(x_batch, state)
    #{'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state['c2'].data.reshape(state['c2'].data.shape[0]*state['c2'].data.shape[1],)


def test():
    import time
    t = "how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow?  \
         how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? \ \
         how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? \
         how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow? how is it going my friend, why are you so slow?"
    c = convert(t)
    s = time.time()
    d = embed_sentence(c)
    print time.time()-s