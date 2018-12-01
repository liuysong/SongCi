#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 用于获取数据，调取模型，以及最后的训练

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data
import numpy as np

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


#读取文件返回data的list形式数据
vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

#获取embeding过程中声称的dict文件
with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

#获取embeding过程中声称的dict文件
with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')

#添加代码获取embeding文件
embed = FLAGS.embeding_npy

#套用现有框架以构建模型
model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build(embedding_file=embed)


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')
    
    #按照循环次数进行训练,应该是需要对model里面的值进行赋值或者其他逻辑处理
    for x in range(1):
        logging.debug('epoch [{0}]....'.format(x))
        state = sess.run(model.state_tensor)
        for dl in utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):

            ##################
            # Your Code here
            ##################
            # 批量读取的数据进行训练
            # 这里处理feed_dict? self.X,self.Y,self.keep_prob需要赋值
            x = dl[0]
            y = dl[1]
            #logging.debug('x:{0} x:{1}'.format(x.shape,x[0]))
            #logging.debug('y:{0} y:{1}'.format(y.shape,y[0]))

            input_x = utils.index_data(x, dictionary)
            #input_x = tf.squeeze(input_x,[0,1])
            lable_y = utils.index_data(y, dictionary)
            #lable_y = tf.squeeze(lable_y,[0,1])

            #logging.debug('input_x:{0}'.format(input_x.shape))
            #logging.debug('lable_y:{0}'.format(lable_y.shape))
            feed_dict = {model.X:input_x,
                    model.Y:lable_y,
                    model.keep_prob:0.7
                    }

            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % 10 == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
