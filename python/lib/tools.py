#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:49:10 2017

@author: zhaohj
"""


import random
import cv2
import numpy as np
import matplotlib.pylab as plt
import copy
from os import walk



def process_ground_truth(bbox):
    """
    Process ground truth box.
    """
    
    # [x1,y1, x2,y2, x3,y3, x4,y4]
    if bbox.shape[1] == 8:
        x1 = bbox[:,0::8]
        y1 = bbox[:,1::8]
        x2 = bbox[:,2::8]
        y2 = bbox[:,3::8]
        x3 = bbox[:,4::8]
        y3 = bbox[:,5::8]
        x4 = bbox[:,6::8]
        y4 = bbox[:,7::8]  

        x_temp = np.hstack((x1,x2,x3,x4))
        y_temp = np.hstack((y1,y2,y3,y4)) 
        
        x = np.min(x_temp,1).reshape(-1,1)
        y = np.min(y_temp,1).reshape(-1,1)
        w = np.max(x_temp,1).reshape(-1,1) - x
        h = np.max(y_temp,1).reshape(-1,1) - y   
    # [x,y,w,h]
    elif bbox.shape[1] == 4:
        x = bbox[:,0::4]
        y = bbox[:,1::4]
        w = bbox[:,2::4]
        h = bbox[:,3::4]
        
    return np.hstack((x,y,w,h))


def generate_samples(img, gtbox, num, gtype, sigma1, sigma2, iou_down, iou_up, clip, display):
    """
    Generate samples.
    """
    bbox = copy.deepcopy(gtbox)
    img_w = img.shape[1]
    img_h = img.shape[0]

    center = np.array([0,0])
    center[0] = bbox[0] + bbox[2]/2
    center[1] = bbox[1] + bbox[3]/2
    
    samples = np.zeros((num,4),dtype='int32')
    
    # sample in whole img, generate neg samples
    if gtype == 'whole':
        samples[:,0] = np.array([random.uniform(0, img_w-bbox[2]) for i in range(num)],dtype='int64')
        samples[:,1] = np.array([random.uniform(0, img_h-bbox[3]) for i in range(num)],dtype='int64')
        
        samples[:,2] = np.array([random.gauss(bbox[2], sigma2) for i in range(num)],dtype='int64')
        samples[:,3] = np.array([random.gauss(bbox[3], sigma2) for i in range(num)],dtype='int64')
    
    # sample around object, generate pos samples
    elif gtype == 'around':
        samples[:,0] = np.array([random.gauss(center[0], sigma1) for i in range(num)],dtype='int64')
        samples[:,1] = np.array([random.gauss(center[1], sigma1) for i in range(num)],dtype='int64')
        
        samples[:,2] = np.array([random.gauss(bbox[2], sigma2) for i in range(num)],dtype='int64')
        samples[:,3] = np.array([random.gauss(bbox[3], sigma2) for i in range(num)],dtype='int64')

        samples[:,0] = samples[:,0] - samples[:,2]/2
        samples[:,1] = samples[:,1] - samples[:,3]/2
    
    # clip boxes 
    if clip:
        samples = clip_boxes(samples, img.shape)
        
    # iou sample
    sample_iou = overlap_ratio(bbox,samples)
    if gtype == 'around':
        samples = samples[sample_iou>iou_down,:]
        sample_iou = overlap_ratio(bbox,samples)
        samples = samples[sample_iou<iou_up,:]
    elif gtype == 'whole':
        samples = samples[sample_iou>iou_down,:]
        sample_iou = overlap_ratio(bbox,samples)
        samples = samples[sample_iou<iou_up,:]

    samples = samples[samples[:,2] > 0,:]
    samples = samples[samples[:,3] > 0,:]
    
    """display boxes"""
    if display:
        im_copy = img.copy()
        for i in range(len(samples)):
            cv2.rectangle(im_copy,
                          (samples[i,0],samples[i,1]),
                          (samples[i,0]+samples[i,2], samples[i,1]+samples[i,3]),
                          (0,255,0), 1)
        plt.imshow(im_copy)
        plt.show()
        

    return samples.astype(int)


def overlap_ratio(bbox, boxes):
    """
    Compute overlap ratio.
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2] - 1
    y2 = bbox[1] + bbox[3] - 1
    
    """compute ground truth area"""
    gtArea = bbox[2]*bbox[3]
    
    boxes = np.reshape(boxes,[-1,4])
    box_num = len(boxes)
    
    
    overlap_ratio = []
    for i in range(box_num):       
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(x1, boxes[i,0])
        yA = max(y1, boxes[i,1])
        xB = min(x2, boxes[i,0]+boxes[i,2]-1)
        yB = min(y2, boxes[i,1]+boxes[i,3]-1)
        
        boxArea = boxes[i,2]*boxes[i,3]

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)
                
        # no overlap
        if (x2 <= boxes[i,0] or y2 <= boxes[i,1]) or ((boxes[i,0]+boxes[i,2]) <= x1 or (boxes[i,1]+boxes[i,3]) <= y1):
            interArea = 0        
        
        # compute IoU
        overlap_ratio.append( interArea / float(float(gtArea) + boxArea - interArea) ) #float(gtArea) int16 to float 64bits 
        
    return np.array(overlap_ratio)


def candidate_region(img, bbox, drift):
    """
    Gennerate candidate region.
    (this function is only used for generate 9 candidate regions)
    """
    
    candidate = np.zeros([9,4],dtype='int32')
    
    w = bbox[2]
    h = bbox[3]
    
    img_w = img.shape[1]
    img_h = img.shape[0]
    
    x_l = bbox[0]-bbox[2]*drift
    x_c = bbox[0]
    x_r = bbox[0]+bbox[2]*drift
    
    y_u = bbox[1]-bbox[3]*drift
    y_c = bbox[1]
    y_d = bbox[1]+bbox[2]*drift
    
    if x_l < 0: x_l = 0
    if y_u < 0: y_u = 0
    
    candidate[0,:] = np.array([x_l,y_u,w,h],dtype='int32')
    candidate[1,:] = np.array([x_c,y_u,w,h],dtype='int32')
    candidate[2,:] = np.array([x_r,y_u,w,h],dtype='int32')
    
    candidate[3,:] = np.array([x_l,y_c,w,h],dtype='int32')
    candidate[4,:] = np.array([x_c,y_c,w,h],dtype='int32')
    candidate[5,:] = np.array([x_r,y_c,w,h],dtype='int32')
    
    candidate[6,:] = np.array([x_l,y_d,w,h],dtype='int32')
    candidate[7,:] = np.array([x_c,y_d,w,h],dtype='int32')
    candidate[8,:] = np.array([x_r,y_d,w,h],dtype='int32')
    
    if (x_r + w) > (img_w-1):
        candidate[2,2] = img_w-1-x_r
        candidate[5,2] = img_w-1-x_r
        candidate[8,2] = img_w-1-x_r
    if (y_d + h) > (img_h-1):
        candidate[2,3] = img_h-1-y_d
        candidate[5,3] = img_h-1-y_d
        candidate[8,3] = img_h-1-y_d
    
    return candidate


def clip_boxes(boxes,im_shape):
    """
    Clip boxes to image boundaries.
    [x,y,w,h]
    """

    
    # if only one box, reshape array from (4,) to (:,4)
    boxes = boxes.reshape(-1,4)   
    
    
    # to [x1,y1,x2,y2]
    boxes[:,2::4] += boxes[:,0::4]
    boxes[:,3::4] += boxes[:,1::4]
    
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    
    # to [x,y,w,h]
    boxes[:,2::4] -= boxes[:,0::4] 
    boxes[:,3::4] -= boxes[:,1::4]
    
    return boxes


def crop_to_bigger(img, box, times):
    """
    Enlarge bbox and crop img according to the bbox.
    """
    
    bbox = copy.deepcopy(box)
    
    # enlarge bbox 3 times
    cx = bbox[0]+bbox[2]/2
    cy = bbox[1]+bbox[3]/2
    bbox[2] *= times
    bbox[3] *= times
    bbox[0] = cx -bbox[2]/2
    bbox[1] = cy -bbox[3]/2

    bbox = clip_boxes(bbox, img.shape).reshape(4)
    
    crop = img[bbox[1]:bbox[1]+bbox[3],
               bbox[0]:bbox[0]+bbox[2],:]

    return crop

    
    

def crop_and_pad(img, x1,y1,x2,y2):
    """
    Crop img patch and pad with mean.
    """
    
    img_w = img.shape[1]
    img_h = img.shape[0]

    # compute mean
    mean = np.array( [np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])], dtype=float )
#    mean = np.zeros(3).astype(int)

    # compute numbers of column or row to be pad    
    pad_left = 0
    pad_right = 0
    pad_up = 0
    pad_down = 0
    if x1 < 0:
        pad_left = int(-x1)
        x1 = 0
    if y1 < 0:
        pad_up = int(-y1)
        y1 = 0
        
    if x2 > img_w:
        pad_right = int(x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        pad_down = int(y2 - img_h)
        y2 = img_h
        
    crop = img[y1:y2,x1:x2,:]
    
    # padding
    if pad_left != 0:
        pad = np.ones([crop.shape[0],pad_left,3])
        pad[:,:] = mean
        crop = np.concatenate((pad,crop),axis=1)
        
    if pad_right != 0:
        pad = np.ones([crop.shape[0],pad_right,3])
        pad[:,:] = mean
        crop = np.concatenate((crop,pad),axis=1)
        
    if pad_up != 0:
        pad = np.ones([pad_up,crop.shape[1],3])
        pad[:,:] = mean
        crop = np.concatenate((pad,crop),axis=0)
    
    if pad_down != 0:
        pad = np.ones([pad_down,crop.shape[1],3])
        pad[:,:] = mean
        crop = np.concatenate((crop,pad),axis=0)
    
    return crop.astype('uint8')
    

def get_video_list(vid_dir, vid_set='vot'):
    """
    Get videos list.
    """
    
    if vid_set == 'vot':
        # read vot list.txt
        with open(vid_dir + '/list.txt','r') as list_file:
            seq_list = list_file.readlines()
            
        # remove '\n'
        for index, seq_i in enumerate(seq_list):
            seq_list[index] = seq_i.splitlines()[0]
            vid_num = len(seq_list)
        
        return seq_list,vid_num

    elif vid_set == 'otb':
        for (dirpath, seq_list, filenames) in walk(vid_dir):
            break
        seq_list.sort()
        vid_num = len(seq_list)
        return seq_list,vid_num

def load_video(vid_dir, vid_list, vid_id, vid_set='vot'):  
    """
    Load video and return first frame\gtBox\frame_num.
    """
    
    if vid_set == 'vot':
        # read ground truth of first video
        with open(vid_dir + '/' + vid_list[vid_id] + '/groundtruth.txt', 'r') as gt_file:
            gt_box = gt_file.readlines()
            frame_num = len(gt_box)
        
        # remove '\n' and convert str list to float array
        for index, gt_box_i in enumerate(gt_box):
            gt_box[index] = np.array(map( float,gt_box_i.splitlines()[0].split(',') ), dtype=int)   
        
        # read first frame (B,G,R)
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/{:0>8d}.jpg'.format(1))
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
        
        return frame,np.array(gt_box),frame_num
        
    elif vid_set == 'otb':
        # read ground truth of first video
        if vid_list[vid_id] != 'Jogging' and vid_list[vid_id] != 'Skating2':
            with open(vid_dir + '/' + vid_list[vid_id] + '/groundtruth_rect.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)
        else:
            with open(vid_dir + '/' + vid_list[vid_id] + '/groundtruth_rect.1.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)
            
        # remove '\n' and convert str list to float array
        for index, gt_box_i in enumerate(gt_box):
            try:
                gt_box[index] = np.array(map( float,gt_box_i.splitlines()[0].split(',') ), dtype=int)   
            except:
                try:
                    gt_box[index] = np.array(map( float,gt_box_i.splitlines()[0].split('\t') ), dtype=int)   
                except:
                    gt_box[index] = np.array(map( float,gt_box_i.splitlines()[0].split(' ') ), dtype=int)   
        # get img list, reason for clutter img name
        for (dirpath, seq_list, filenames) in walk(vid_dir + '/' + vid_list[vid_id] + '/img/'):
            break
        filenames.sort()
        # read first frame (B,G,R)
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[0])
        # add channels for single channel gray img
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
    
        return frame,np.array(gt_box),frame_num

def load_video_frame(vid_dir, vid_list, vid_id, frame_id, vid_set='vot'):
    """
    Load frame of video.
    """
    
    if vid_set == 'vot':
        # read first frame (B,G,R)
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/{:0>8d}.jpg'.format(frame_id))
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
        
        return frame
        
    elif vid_set == 'otb':
        # get img list, reason for clutter img name
        for (dirpath, seq_list, filenames) in walk(vid_dir + '/' + vid_list[vid_id] + '/img/'):
            break
        filenames.sort()
        # read first frame (B,G,R)
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1])
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
    
        return frame

# %%
    
if __name__ == "__main__":

    print "This is a module for tracker."
