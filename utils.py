#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np

#读入文本信息,将文件内容转换为list时,相当于将所有的内容当做字符串转换为list,将每个汉字会作为一个list元素生成整个list
def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)

# 获取训练数据,需要返回批量形式的数据
def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    raw_x = vocabulary
    raw_y = vocabulary[1:]
    
    data_length = len(raw_x)
    batch_partition_length = data_length // batch_size
    data_x = np.empty([batch_size, batch_partition_length], dtype=np.str)
    data_y = np.empty([batch_size, batch_partition_length], dtype=np.str)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
