import pdb
import numpy as np
import os



with open('data/split1.train', 'r') as f:
    video_list1 = f.read().split('\n')[0:-1]

with open('data/split1.test', 'r') as f:
    video_list2 = f.read().split('\n')[0:-1]


for video in video_list1:
    z = np.random.rand(512)
    path = os.path.join('data', 'hyperfeatures')
    name = video + '.npy'
    np.save(os.path.join(path, name), z)

for video in video_list2:
    z = np.random.rand(512)
    path = os.path.join('data', 'hyperfeatures')
    name = video + '.npy'
    np.save(os.path.join(path, name), z)
