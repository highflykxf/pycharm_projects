import keras
import pandas as pd
import pickle
from datetime import  datetime, timedelta
import numpy as np
from collections import defaultdict
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding,Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations,regularizers,constraints
from keras.optimizers import SGD, RMSprop, Adagrad


dir = 'data_sample/'
max_len = 100

class AttLayer(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],input_shape[-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.uw = self.add_weight((input_shape[-1],),
                                  initializer=self.init,
                                  name='{}_u'.format(self.name))
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # print(x.shape)
        eij = K.dot( x, self.W)
        # print(eij.shape)
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        # print(eij.shape)
        eij = K.dot(eij, self.uw )
        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

class Data(object):
    def __init__(self):
        self.starttime = datetime.strptime('2016-02-01','%Y-%m-%d')
        pass

    def read_file(self, fname):
        user_ids = self.user_ids
        series = self.series
        actions = pd.read_csv(fname)
        last_user_id = -1
        starttime = self.starttime
        for index, row in actions.iterrows():
            user_id = int( row['user_id'] )
            if user_id != last_user_id:
                last_user_id = user_id
                user_index = 0
                if 'U_'+str(user_id) in self.all_dict:
                    user_index = self.all_dict['U_'+str(user_id)]
                user_ids.append(user_index)
                series.append( list() )
            sku_id = int(row['sku_id'])
            type = int(row['type'])
            #没有学习删除购物车
            if type == 3:
                continue
            time = row['time']
            sku_index = 0
            if 'S_'+str(sku_id) in self.all_dict:
                sku_index = self.all_dict['S_'+str(sku_id)]
            type_index = self.all_dict['actions_'+str(type)]
            t = (datetime.strptime(time,'%Y-%m-%d %H:%M:%S')-starttime).seconds / 3600
            series[-1].append( [type_index, sku_index, t ] )


    def load_data(self):
        self.all_dict = pickle.load( open(dir + 'attr_triples.dict', 'rb'))
        fnames = ['compares_{}.csv'.format(sep) for sep in range(10)]

        self.user_ids = []
        self.series = []
        for fname in fnames:
            self.read_file(dir + fname)
            #先测试一个文件
            break


    def get_label(self, u_series, si, t):
        label = 0
        for i in range(si, len(u_series)):
            ti = u_series[i][2]
            type_index = u_series[i][0]
            if ti > t+ 5*24:
                break
            if type_index == self.buy_index:
                label = 1
                break
        return label

    def pad(self, u_series, max_len ):
        pad_tuple = [0, 0, -100]
        series = []
        if len(u_series) >= max_len:
            series.extend( u_series[-max_len:])
        else:
            for i in range(max_len-len(u_series)):
                series.append( pad_tuple )
            series.extend( u_series )
        return series

    def gen_dataset(self):
        self.buy_index = self.all_dict['actions_4']
        train_start = (datetime.strptime( '2016-04-01','%Y-%m-%d') - self.starttime).seconds/3600
        #用于线下调试的训练
        train_valid_end = (datetime.strptime( '2016-04-06','%Y-%m-%d') - self.starttime).seconds/3600
        #11-16用于表示线下结果
        valid_start = (datetime.strptime( '2016-04-11','%Y-%m-%d') - self.starttime).seconds/3600
        user_ids = self.user_ids
        series = self.series
        #(N,T,4) 4:type, sku_id, time
        #全部调试数据
        train_all = []
        train_all_userids = []
        train_all_labels = []
        #本地调试的训练数据
        train_valid = []
        train_valid_userids = []
        train_valid_labels = []
        #产生验证结果
        test_valid = []
        test_valid_userids = []
        #生成提交结果
        test = []
        test_userids = []

        true_max_len = 0
        for i in range( len( user_ids )):
            u_series = series[i]
            for si in range( len( u_series ) ):
                tuple = u_series[si]
                type_index, sku_index, t = tuple[0], tuple[1], tuple[2]
                if t >= train_start:
                    pad_series = self.pad(u_series[0:si+1], max_len)
                    label = self.get_label(u_series, si, t)
                    if t<=valid_start:
                        train_all_labels.append(label)
                        train_all.append( pad_series )
                        train_all_userids.append( user_ids[i] )
                    if t<= train_valid_end:
                        train_valid_labels.append( label )
                        train_valid.append( pad_series )
                        train_valid_userids.append( user_ids[i] )
                    elif t<valid_start and t>train_valid_end:
                        test_valid.append( pad_series )
                        test_valid_userids.append( user_ids[i] )
                    elif t >= valid_start:
                        test.append( pad_series )
                        test_userids.append( user_ids[i] )

                    if si+1 > true_max_len:
                        true_max_len = si+1
        print('maxlen: {}'.format(true_max_len) )
        print('train all shape: {}'.format( len(train_all) ) )
        train_all_dict = dict()
        train_all_dict['train'] = np.array( train_all )
        train_all_dict['label'] = np.array( train_all_labels )
        train_all_dict['userids'] = np.array(train_all_userids )
        self.save(train_all_dict, 'cache/train_all.dict')
        train_valid_dict = dict()
        train_valid_dict['train' ] = np.array( train_valid )
        train_valid_dict['label'] = np.array( train_valid_labels )
        train_valid_dict['userids'] = np.array( train_valid_userids )
        self.save( train_valid_dict, 'cache/train_valid.dict')

        test_valid_dict = dict()
        test_valid_dict['train'] = np.array( test_valid )
        test_valid_dict['userids'] = np.array(test_valid_userids )
        self.save( test_valid_dict, 'cache/test_valid.dict')

        test_dict = dict()
        test_dict['train'] = np.array( test )
        test_dict['userids'] = np.array( test_userids )
        self.save( test_dict, 'cache/test.dict')

    def save(self, s_dict, fname ):
        pickle.dump( s_dict, open(fname, 'wb' ) )


class SeriesModel(object):
    def __init__(self):
        pass
        # self.train_all_dict = self.load('cache/train_all.dict')
        # self.train_valid_dict = self.load( 'cache/train_valid.dict')
        # self.test_valid_dict = self.load('cache/test_valid.dict')
        # self.test_dict = self.load( 'cache/test.dict')

    def load(self, fname):
        return pickle.load( open(fname, 'rb'))

    def define(self ):
        mat = np.loadtxt(dir+'attrs_vecs.txt')
        embedding_layer = Embedding(mat.shape[0],
                                    mat.shape[1],
                                    weights=[mat],
                                    input_length=2,
                                    mask_zero=False,
                                    trainable=False)
        embedding_layer2 = Embedding(mat.shape[0],
                                    mat.shape[1],
                                    weights=[mat],
                                    input_length=1,
                                    mask_zero=False,
                                    trainable=False)
        input = Input(shape=(2,), dtype = 'int32')
        embeded_unit = embedding_layer( input )
        flat = Flatten()(embeded_unit)
        unit_encoder = Model( input, flat)
        series_input = Input( shape=(max_len, 2), dtype='int32')
        encoded = TimeDistributed(unit_encoder)(series_input)
        l_lstm = Bidirectional( GRU( 50, return_sequences=True ) )(encoded)
        l_lstm = Masking( )(l_lstm)
        l_att = AttLayer()(l_lstm)
        user_input = Input( shape=(1,), dtype='int32')
        embeded_user = embedding_layer2( user_input )
        embeded_user = Flatten()(embeded_user)
        merged = Merge(mode='concat', concat_axis=1)( [l_att, embeded_user])
        pred = Dense(1, activation='sigmoid' )(merged)
        model = Model( [series_input, user_input], pred )
        lr = 1e-3
        opt = Adagrad(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['acc'] )
        return model

    def tune(self):
        model = self.define( )
        train_valid_dict = self.load('cache/train_valid.dict')
        test_valid_dict = self.load('cache/test_valid.dict')
        y = train_valid_dict['label']
        userids = train_valid_dict['userids']
        userids.dtype = np.int32
        x = train_valid_dict['train']
        print(x.shape)
        x = x[:,:,0:2]
        x.dtype = np.int32
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=1)
        save_best = ModelCheckpoint('./cache/gru_model.hdf5', save_best_only=True)
        model.fit([x, userids], y,
                  validation_split=0.2, callbacks=[early_stopping, save_best],
                  nb_epoch=20, batch_size=128)

        x_val = test_valid_dict['train']
        x_val = x_val[:,:,0:2]
        x_val.dtype = np.int32
        userids_val = test_valid_dict['userids']
        y_hat = model.predict( [x_val, userids_val], batch_size= 128 )
        df = np.array( (y_hat.shape[0],2) )
        df[:,0 ] = userids_val
        df[:,1 ] = y_hat
        df = pd.DataFrame( data = df, columns=['user_id','label'])
        import sep_model_learn
        sep_model_learn.sep_model_performance( df )

    def sub(self):
        self.train_all_dict = self.load('cache/train_all.dict')
        self.test_valid_dict = self.load('cache/test_valid.dict')
        self.test_dict = self.load('cache/test.dict')
        #TODO 生成提交结果

if __name__ == '__main__':
    fnames = ['compares_{}.csv'.format(sep) for sep in range(10)]
    # data = Data( )
    # data.load_data( )
    # print('load data done.')
    # data.gen_dataset( )
    # print('done.')
    model = SeriesModel()
    model.tune( )