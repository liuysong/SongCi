### 第十一周作业

- embeding数据处理
  - 详见代码https://github.com/liuysong/AIliuys/blob/master/20180925_week11/quiz-w10-code/word2vec_basic_ch.py
  - 在原代码基础上，添加了matplotlib对中文的支持，并且同时保存了embeding的词向量矩阵，以及dictionay和reverse_dictionary
- 模型代码修改
  - 详见https://github.com/liuysong/AIliuys/tree/master/20180925_week11/quiz-w10-code
  - 除了修改指定代码之外，添加了embeding_npy变量，以获取word2文件所生成的embeding文件用于训练当中。
- 关于rnn网络，主要有两个参数花了一些时间去理解
  - 关于rnn_layers参数的含义，课程中的代码复现的应该都是1层RNN网络，即不考虑上一个state到下一个state之间的结构的话，一个cell中从输入到输出之间的网络层数。
  - 关于num_steps参数的含义，相当于每次训练，state的循环进行num_step次
- 模型训练的日志和结果
  - 详见https://www.tinymind.com/executions/rcn8a13z
