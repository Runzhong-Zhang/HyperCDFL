#!/usr/bin/python3.7

import numpy as np
from utils.dataset import Dataset
from utils.network import Trainer, Forwarder
from utils.viterbi import Viterbi
from torch.utils.tensorboard import SummaryWriter
import os
import pdb
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
writer = SummaryWriter()
### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open('data/mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]

### read training data #########################################################
with open('data/split1.train', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = Dataset('data', video_list, label2index, shuffle = True)

### generate path grammar for inference ########################################
paths = set()
for _, transcript, _ in dataset:
    paths.add( ' '.join([index2label[index] for index in transcript]) )
with open('results/grammar.txt', 'w') as f:
    f.write('\n'.join(paths) + '\n')

### actual nn-viterbi training #################################################
decoder = Viterbi(None, None, frame_sampling = 30) # (None, None): transcript-grammar and length-model are set for each training sequence separately, see trainer.train(...)
trainer = Trainer(decoder, dataset.input_dimension, dataset.n_classes, buffer_size = len(dataset), buffered_frame_ratio = 25)
learning_rate = 0.01
window = 10
step = 5


# train for 100000 iterations
for i in range(100000):
    # print(i)
    sequence, transcript, z_specific = dataset.get()
    loss1, loss2 = trainer.train(sequence, transcript, z_specific, batch_size=512, learning_rate=learning_rate, window=window, step=step)
    # print some progress information
    if (i+1) % 100 == 0:
        print('Iteration %d, loss1: %f, loss2: %f, loss: %f' % (i+1, loss1, loss2, loss1 - loss2))
        writer.add_scalar('loss1', loss1, i)
        writer.add_scalar('loss2', loss2, i)
        writer.add_scalar('loss1-loss2', loss1-loss2, i)
    # save model every 1000 iterations
    if (i+1) % 1000 == 0:
        network_file = 'results/network.iter-' + str(i+1) + '.net'
        length_file = 'results/lengths.iter-' + str(i+1) + '.txt'
        prior_file = 'results/prior.iter-' + str(i+1) + '.txt'
        trainer.save_model(network_file, length_file, prior_file)
    # adjust learning rate after 60000 iterations
    if (i+1) == 60000:
        learning_rate = learning_rate * 0.1

writer.close()