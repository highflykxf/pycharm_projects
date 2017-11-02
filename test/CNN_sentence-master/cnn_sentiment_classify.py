# -*- coding:utf-8 -*-
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time
warnings.filterwarnings("ignore")
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
import theano
from keras.optimizers import Adadelta
np.random.seed(2)
theano.config.compute_test_value = 'warn'


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=56, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]

if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W

    # Model Hyperparameters
    sequence_length = 64
    embedding_dim = 300
    filter_sizes = (3, 4, 5)
    num_filters = 32
    dropout_prob = (0.5, 0.5)
    hidden_dims = 100

    # Training parameters
    batch_size = 50
    num_epochs = 100
    val_split = 0.1

    results = []
    for i in range(0,10):
        #构建交叉验证数据集
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=56, k=300, filter_h=5)
        # Building model
        # ==================================================
        #
        # graph subnet with one input and one output,
        # convolutional layers concateneted in parallel
        graph_in = Input(shape=(sequence_length, embedding_dim))
        convs = []
        for fsz in filter_sizes:
            conv = Convolution1D(nb_filter=num_filters,
                                 filter_length=fsz,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1)(graph_in)
            pool = MaxPooling1D(pool_length=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)

        if len(filter_sizes) > 1:
            out = Merge(mode='concat')(convs)
        else:
            out = convs[0]

        graph = Model(input=graph_in, output=out)

        # main sequential model
        model = Sequential()
        if non_static == True:
            model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length))
        model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
        model.add(graph)
        model.add(Dense(hidden_dims))
        model.add(Dropout(dropout_prob[1]))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adadelta(lr=0.05), metrics=['accuracy'])
        model.summary()
        graph.summary()
        # Training model
        # ==================================================
        model.fit(datasets[0][:,:sequence_length], datasets[0][:,-1], batch_size=batch_size,
                  nb_epoch=num_epochs, verbose=2,
                  validation_data=(datasets[1][:,:sequence_length], datasets[1][:,-1]))
