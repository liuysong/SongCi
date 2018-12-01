#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf
import numpy as np

import utils
#from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()

#vocabulary = read_data(FLAGS.text)
#print('Data size', len(vocabulary))

#with open(FLAGS.text,encoding="utf-8") as f:
#    data=f.read()
#data = list(data)
#print (len(data))
#print (data[0:5])


#embedding = np.load('../final_embedfiles')
#embed = tf.constant(embedding,name='embedding')

#print (embed.shape)

#data = tf.nn.embedding_lookup(embed,[7,3])
#print (data)
#print (data.shape)
#with tf.Session():
#    print (data.eval())

#for dl in utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
    #print (len(dl))
#    print ('x:{0}'.format(dl[0]))
#    print ('y:{0}'.format(dl[1]))


a=[[[1,2,3],
    [4,5,6],
   [7,8,9]],
   [[10,20,30],
    [40,50,60],
    [70,80,90]]
   ]
b=tf.constant(a)

print ( b.shape )

c=tf.concat(b,1)
print ( c.shape )
with tf.Session() as se:
    print ( c.eval() )
