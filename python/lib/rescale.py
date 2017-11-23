

import sys

sys.path.append('/home/zhaohj/Public/caffe-faster-rcnn/python')
import caffe

sys.path.append('/home/zhaohj/Documents/Workspace/RPNT_py/lib')
import tools
import rpn

import cv2
import numpy as np
import matplotlib.pylab as plt
import copy
import random
import math

# %%

class ReScaleLayer(caffe.Layer):

    
    def setup(self, bottom, top):
        self._in_map = bottom[0].data[0]
        
    def reshape(self, bottom, top):
        top[0].reshape(1,
                       bottom[0].data.shape[1],
                       bottom[0].data.shape[2],
                       bottom[0].data.shape[3])
        
        
    def forward(self, bottom, top):

        in_map = bottom[0].data[0].astype(float)
        num_channel = bottom[0].data.shape[1] # num of channel

        for c in range(num_channel):
            if ( np.max(in_map[c]) - np.min(in_map[c]) ) != 0:
                top[0].data[0][c] = ( in_map[c] - np.min(in_map[c]) ) / ( np.max(in_map[c]) - np.min(in_map[c]) ) * 255
            else:
                top[0].data[0][c] = in_map[c]

            # top[0].data[0][c] = in_map[c]
        
    def backward(self, bottom, top, propagate_down):
        pass