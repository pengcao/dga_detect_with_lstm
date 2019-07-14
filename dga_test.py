#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2016-2020, Big Bird Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: dga_test.py
@time: 2019/7/11 22:38
@desc:
'''
from model.LstmModel import LstmModel
import codecs
import numpy as np
import datetime

if __name__ == '__main__':
    
    batch_size = 100 # 批处理大小
    epochs = 1       # 训练轮数
    trainDataPath = './test_data/test_data_domain.txt'  # 原始数据文件路径
    modelPath = 'dga_by_lstm_model-attend0712.h5'  # 模型文件保存路径或读取路径
    resultPath = './test_data/test_data_domainAttention_result-0712.txt'

    # 读取配置文件
    charList = {}
    confFilePath = './conf/charList.txt'
    confFile = codecs.open(filename=confFilePath, mode='r', encoding='utf-8', errors='ignore')
    lines = confFile.readlines()
    # 字符序列要从1开始,0是填充字符
    ii = 1
    for line in lines:
        temp = line.strip('\n').strip('\r').strip(' ')
        if temp != '':
            charList[temp] = ii
            ii += 1

    max_features = ii
    #
    #  训练数据
    # 转换数据格式
    x_data_sum = []
    trainFile = codecs.open(filename=trainDataPath, mode='r', encoding='utf-8', errors='ignore')
    lines = trainFile.readlines()
    for line in lines:
        if line.strip('\n').strip('\r').strip(' ') == '':
            continue

        x_data = []
        x = line.strip('\n').strip('\r').strip(' ')

        for char in x:
            try:
                x_data.append(charList[char])
            except:
                print('unexpected char' + ' : ' + char)
                x_data.append(0)

        x_data_sum.append(x_data)

    x_data_sum = np.array(x_data_sum)

    # LstmModel
    lstmModel = LstmModel()
    lstmModel.predict(x_data_sum, batch_size, modelPath, resultPath)