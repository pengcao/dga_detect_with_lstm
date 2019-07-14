#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2016-2020, Big Bird Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: LstmWithAttentionModel.py
@time: 2019/7/10 22:30
@desc:
'''

from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import multiply
from keras.layers.core import *
from keras.models import *

class LstmWithAttentionModel(object):

    def __init__(self):
        pass

    def linear_attention(self,inputs,item_steps):
        '''

        :param inputs:
        :param item_steps:
        :return:
        '''
        p_inputs = Permute((2, 1))(inputs) # 128*75
        d_inputs = Dense(item_steps, activation='softmax')(p_inputs) # 128*75
        a_probs = Permute((2, 1), name='attention_vec')(d_inputs)
        # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def nonlinear_attention_relu(self,inputs,item_steps):
        '''

        :param inputs:
        :param item_steps:
        :return:
        '''
        p_inputs = Permute((2, 1))(inputs)
        d_inputs = Dense(item_steps, activation='softmax')(p_inputs)
        d_inputs = Dense(30, activation='relu')(d_inputs)
        d_inputs = Dense(item_steps, activation='softmax')(d_inputs)
        a_probs = Permute((2, 1), name='attention_vec')(d_inputs)
        # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def nonlinear_attention_tanh(self,inputs,item_steps):
        '''

        :param inputs:
        :param item_steps:
        :return:
        '''
        # Permute层将输入的维度按照给定模式进行重排,
        # (2, 1)将输入的第二个维度重拍到输出的第一个维度，而将输入的第一个维度重排到第二个维度
        p_inputs = Permute((2, 1))(inputs)
        #  stack a deep densely-connected network on top
        d_inputs = Dense(item_steps, activation='softmax')(p_inputs)
        d_inputs = Dense(30, activation='tanh')(d_inputs)
        d_inputs = Dense(item_steps, activation='softmax')(d_inputs)
        a_probs = Permute((2, 1), name='attention_vec')(d_inputs)
        # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        # Multiply的函数式包装
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def model_lstm_with_attention(self,shape, max_features):
        '''

        :param shape:
        :param max_features:
        :return:
        '''
        inputs = Input(shape=(shape,))
        # This embedding layer will encode the input sequence
        # into a sequence of dense 128-dimensional vectors.
        e_inputs = Embedding(max_features, 128, input_length=shape)(inputs)
        lstm_out = LSTM(128, return_sequences=True)(e_inputs)  # 返回值 (item_steps, input_dim) = (75,128)
        #
        attention_mul = self.nonlinear_attention_tanh(lstm_out, 75)
        attention_mul = Flatten()(attention_mul)    # 折叠成一维向量
        d_inputs = Dropout(0.5)(attention_mul)
        d_inputs = Dense(1)(d_inputs)
        outputs = Activation('sigmoid')(d_inputs)
        model = Model(input=[inputs], output=outputs)

        return model

    def trainModel(self,max_features, x_train, y_train, batch_size, epochs, model_path):
        '''
        :param max_features:
        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :param model_ath:
        :return:
        '''

        x_train = sequence.pad_sequences(x_train, maxlen=75)
        model = self.model_lstm_with_attention(75, max_features)

        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

        model.save(model_path)

    def predict(self,x_test, batch_size, model_path, result_path):
        '''

        :param x_test:
        :param batch_size:
        :param model_path:
        :param result_path:
        :return:
        '''
        x_test = sequence.pad_sequences(x_test, maxlen=75)
        model = load_model(model_path)
        y_test = model.predict(x_test, batch_size=batch_size).tolist()

        file = open(result_path, 'w+')
        for index in y_test:
            y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
            file.write(str(y) + '\n')