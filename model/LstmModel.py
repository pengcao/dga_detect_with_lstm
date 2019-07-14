#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2016-2020, Big Bird Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: LstmModel.py
@time: 2019/7/10 22:03
@desc:
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model

class LstmModel(object):
    
    def __init__(self):
        pass

    def trainModel(self,max_features, x_train, y_train, batch_size, epochs, modelPath):
        '''

        :param max_features:
        :param x_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :param modelPath:
        :return:
        '''
        model=Sequential()
        model.add(Embedding(max_features,128,input_length=75))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        # 模型训练
        model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        # 序列的最大长度为75,大于此长度的序列将被截短，小于此长度的序列将在后部填0
        x_train=sequence.pad_sequences(x_train,maxlen=75)
    
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        model.save(modelPath)
    
    def predict(self,x_test, batch_size, modelPath, resultPath):
        '''

        :param x_test:
        :param batch_size:
        :param modelPath:
        :param resultPath:
        :return:
        '''
        x_test = sequence.pad_sequences(x_test, maxlen=75)
        model = load_model(modelPath)
        y_test = model.predict(x_test, batch_size=batch_size).tolist()
    
        file = open(resultPath, 'w+')
        for index in y_test:
            y = float(str(index).strip('\n').strip('\r').strip(' ').strip('[').strip(']'))
            file.write(str(y) + '\n')