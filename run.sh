#!/bin/bash

python train_eval.py --output_dir='./output' --text='./QuanSongCi.txt' --num_steps=5 --batch_size=200 --dictionary='./dictionary.json' --reverse_dictionary='./reverse_dictionary.json' --learning_rate=0.01 --embeding_npy='./embedding.npy'
