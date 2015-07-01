#!/usr/bin/env python

"""
Convolutionary neural network for language, trained on sentiment mining problem
"""

# imports
import argparse
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import cPickle as pkl
import gc, sys, pdb
import bloscpack as bp


# Parsing
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# Paths
data_path_train_x = './data/blp/train_x.blp'
data_path_train_y = './data/blp/train_y.blp'
data_path_test_x = './data/blp/test_x.blp'
data_path_test_y = './data/blp/test_y.blp'

# Hyperparameters
batchsize = 100
n_epoch   = 100

# Load train data
print "\nunpickling training data..."
x_train = bp.unpack_ndarray_file(data_path_train_x)
x_train = x_train[:np.floor(x_train.shape[0]/2.0), :, :]
y_train = bp.unpack_ndarray_file(data_path_train_y)
y_train = y_train[:np.floor(y_train.shape[0]/2.0)]
gc.collect()

# Load test data
print "\nunpickling testing data..."
x_test = bp.unpack_ndarray_file(data_path_test_x)
y_test = bp.unpack_ndarray_file(data_path_test_y)
N_test = x_test.shape[0]
gc.collect()

# data_info
print "\nreading info..."
N = x_train.shape[0]
dim = x_train.shape[1]
max_len = x_train.shape[2]

# reshape data to match chainer format
x_train = np.reshape(x_train, (x_train.shape[0], 1, 300, max_len))

# Compute network dimensions
print "\nplanning dimension of the neural network..."
h_1 = dim
w_1 = max_len
ch_in_1 = 1
ch_out_1 = 6
filter_1 = (1,7)
stride_1 = (1,1)
padding_1 = (0,6)

max_pool_window_1 = (1,11)
max_pool_stride_1 = (1,5)

h_2_beforemax = ((h_1+2*padding_1[0]-filter_1[0])/stride_1[0]+1)
h_2 = ((h_1+2*padding_1[0]-filter_1[0])/stride_1[0]+1)/max_pool_stride_1[0]
w_2_beforemax = ((w_1+2*padding_1[1]-filter_1[1])/stride_1[1]+1)
w_2 = ((w_1+2*padding_1[1]-filter_1[1])/stride_1[1]+1)/max_pool_stride_1[1]

ch_in_2 = ch_out_1
ch_out_2 = 12
filter_2 = (1,5)
stride_2 = (1,1)
padding_2 = (0,4)

avg_pool_window_2 = (2,1)
avg_pool_stride_2 = (2,1)

max_pool_window_2 = (1,7)
max_pool_stride_2 = (1,4)

#h_3 = (((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)/avg_pool_stride_2[0])/max_pool_stride_1[0]
h_3_beforeavg = ((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)
h_3_beforemax = (((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)/avg_pool_stride_2[0])
h_3 = (((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)/avg_pool_stride_2[0])/max_pool_stride_2[0]
w_3_beforeavg = ((w_2+2*padding_2[1]-filter_2[1])/stride_2[1]+1)
w_3_beforemax = (((w_2+2*padding_2[1]-filter_2[1])/stride_2[1]+1)/avg_pool_stride_2[1])
w_3 = (((w_2+2*padding_2[1]-filter_2[1])/stride_2[1]+1)/avg_pool_stride_2[1])/max_pool_stride_2[1]

linear_in_3 = ch_out_2*h_3*w_3
linear_out_3 = 5

print "h_1: ",h_1
print "w_1: ",w_1
print "h_2_beforemax: ",h_2_beforemax
print "h_2: ",h_2
print "w_2_beforemax: ",w_2_beforemax
print "w_2: ",w_2
print "h_3_beforeavg: ",h_3_beforeavg
print "h_3_beforemax: ",h_3_beforemax
print "h_3: ",h_3
print "w_3_beforeavg: ",w_3_beforeavg
print "w_3_beforemax: ",w_3_beforemax
print "w_3: ",w_3
print "lin_in_3: ",linear_in_3

# Define elements of the deep convolutionary network
model = FunctionSet(l1=F.Convolution2D(ch_in_1, ch_out_1, filter_1, stride=stride_1, pad=padding_1),
                    l2=F.Convolution2D(ch_out_1, ch_out_2, filter_2, stride=stride_2, pad=padding_2),
                    l3=F.Linear(linear_in_3, linear_out_3))

# Setup GPU if required
if args.gpu >= 0:
    print "\ninitializing graphical processing unit..."
    cuda.init(args.gpu)
    model.to_gpu()


def evaluate_results(x_test, y_test, N_test, batchsize, max_len):
    '''
    Evaluate model on test set.
    :param x_test:
    :param y_test:
    :param N_test:
    :param batchsize:
    :param max_len:
    :return:
    '''

    # reshape data to match chainer format
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 300, max_len))

    # evaluation
    sum_accuracy = 0
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    print 'test mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test)
    return sum_accuracy / N_test

def forward(x_data, y_data, train=True):
    '''
    Neural net architecture
    :param x_data:
    :param y_data:
    :param train:
    :return:
    '''
    x, t = Variable(x_data), Variable(y_data)

    # h1 = F.dropout(F.leaky_relu(model.l1(x)),  train=train)
    h1 = F.relu(model.l1(x))
    h1 = F.max_pooling_2d(h1,max_pool_window_1,stride=max_pool_stride_1)

    h2 = F.dropout(F.relu(model.l2(h1)))
    h2 = F.average_pooling_2d(h2, avg_pool_window_2, stride=avg_pool_stride_2)
    # h2 = F.local_response_normalization(F.max_pooling_2d(h2,max_pool_window_2,stride=max_pool_stride_2),n=2)
    h2 = F.max_pooling_2d(h2,max_pool_window_2,stride=max_pool_stride_2)

    y  = model.l3(h2)

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

# use momentum SGD first
# is_sgd = False

try:
    max_accuracy = 0
    max_accuracy_epoch = 0
    # Learning loop
    print "\nstart training..."
    for epoch in xrange(0, n_epoch):
        print '\nepoch', epoch + 1

        # dynamic switching of optimizers
        # if epoch % 10 == 0:
        #     if is_sgd:
        #         optimizer = optimizers.AdaGrad()
        #         optimizer.setup(model.collect_parameters())
        #         is_sgd = False
        #     else:
        #         optimizer = optimizers.Adam()
        #         optimizer.setup(model.collect_parameters())
        #         is_sgd = True

        # Training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in xrange(0, N, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()
            loss, acc = forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()

            sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
            sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print 'train mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_accuracy / N)

        eval_accuracy = evaluate_results(x_test, y_test, N_test, batchsize, max_len)
        if eval_accuracy > max_accuracy:
            max_accuracy = eval_accuracy
            max_accuracy_epoch = epoch + 1

    print '\n Max accuracy was {} in epoch {}'.format(max_accuracy, max_accuracy_epoch)

except KeyboardInterrupt:

    evaluate_results(x_test, y_test, N_test, batchsize, max_len)
    print '\n Max accuracy was {} in epoch {}'.format(max_accuracy, max_accuracy_epoch)
    sys.exit(0)
