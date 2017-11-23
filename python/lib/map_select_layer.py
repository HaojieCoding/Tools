

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

class MapSelectLayer(caffe.Layer):

    
    def setup(self, bottom, top):
        self._bbox = bottom[1].data.reshape(4)
        self._in_map = bottom[0].data[0]
        self._side = bottom[0].data.shape[2]
        
    def reshape(self, bottom, top):
        top[0].reshape(1,
                        bottom[0].data.shape[1],
                        bottom[0].data.shape[2],
                        bottom[0].data.shape[3])

        top[1].reshape(1,
                bottom[0].data.shape[1],
                bottom[0].data.shape[2],
                bottom[0].data.shape[3])



            
        
    def forward(self, bottom, top):

        factor = bottom[0].data.shape[2] / 224.0
        bbox = bottom[1].data.reshape(4) * factor
        bbox = bbox.astype(int)

        in_map = bottom[0].data[0].astype(float)

        num_channel = bottom[0].data.shape[1] # num of channel
        ratio = np.zeros(num_channel) 

        for c in range(num_channel):
            act_sum = np.sum(in_map[c])
            act_roi = np.sum(in_map[c][ bbox[1]:bbox[1]+bbox[3],
                                        bbox[0]:bbox[0]+bbox[2] ])
            if act_sum != 0:
                ratio[c] = act_roi/act_sum
            else:
                ratio[c] = 0

        sort = np.argsort(-ratio)

        for c in range(num_channel):
            if np.max(in_map[sort[c]])-np.min(in_map[sort[c]]) != 0:
                map_temp = ( in_map[sort[c]]-np.min(in_map[sort[c]]) ) / ( np.max(in_map[sort[c]])-np.min(in_map[sort[c]]) )*255
                top[0].data[0][c] = map_temp #* (num_channel-c)/num_channel
            else:
                top[0].data[0][c] = in_map[sort[c]]
        

        
    def backward(self, bottom, top, propagate_down):
        pass

        # if bottom[0].data.shape[1]/64 == 8:
        #     bottom[0].diff[0] = np.vstack((top[0].diff[0], top[0].diff[0], top[0].diff[0], top[0].diff[0],
        #                                    top[0].diff[0], top[0].diff[0], top[0].diff[0], top[0].diff[0]))
        # elif bottom[0].data.shape[1]/64 == 4:
        #     bottom[0].diff[0] = np.vstack((top[0].diff[0], top[0].diff[0], top[0].diff[0], top[0].diff[0]))



