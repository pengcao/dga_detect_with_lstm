#!/usr/bin/env python
# encoding: utf-8
'''
@author: caopeng
@license: (C) Copyright 2016-2020, Big Bird Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: data_compare.py
@time: 2019/7/11 23:17
@desc:
'''
import codecs

def getXPredictLabelList(filePath):
    fileReadObj = codecs.open(filePath, 'r', encoding='utf8')
    lines = fileReadObj.readlines()
    labelList = []
    for line in lines:
        if line.strip('\n').strip('\r').strip(' ') == '':
            continue
        s = line.strip('\n').strip('\r').strip(' ').split(' ')
        x = float(s[0])
        if x>0.5:
            labelList.append(1)
        else:
            labelList.append(0)
    return labelList

def getXLabelList(filePath):
    fileReadObj = codecs.open(filePath, 'r', encoding='utf8')
    lines = fileReadObj.readlines()
    labelList = []
    for line in lines:
        if line.strip('\n').strip('\r').strip(' ') == '':
            continue
        s = line.strip('\n').strip('\r').strip(' ').split(' ')
        x = int(s[0])
        labelList.append(x)
    return labelList

if __name__ == '__main__':

    filePredictPath = './test_data/test_data_domainAttention_result-0712.txt'
    fileLabelPath = './test_data/test_data_domain_label.txt'
    filePredictList = getXPredictLabelList(filePredictPath)
    fileLabelList = getXLabelList(fileLabelPath)
    size = len(filePredictList)
    equalSize = 0
    for ii in range(size):
        if filePredictList[ii] == fileLabelList[ii]:
            equalSize = equalSize + 1
    print(' === size : ' , size)
    print(' === equalSize : ', equalSize)
    print(1.0*equalSize/size)
