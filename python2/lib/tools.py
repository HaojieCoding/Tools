#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:49:10 2017

@author: zhaohj
"""

from __future__ import print_function

import random
import cv2
import numpy as np
from os import walk
import matplotlib.pyplot as plt

def gauss_map(cx,cy,w,h,img_w,img_h):

    if w == 0 or h == 0:
        return np.zeros([img_h,img_w])
    
    mu_w = cx
    mu_h = cy
    
    # [-n sigma, +n sigma]
    sigma_w = w*0.2
    sigma_h = h*0.2
#    sigma_w = 2
#    sigma_h = 2
    
    x_w = np.linspace(0,img_w-1,img_w)
    x_h = np.linspace(0,img_h-1,img_h)
    
    y_w = np.exp(-((x_w - mu_w)**2)/(2*sigma_w**2))/(sigma_w * np.sqrt(2*np.pi))
    y_h = np.exp(-((x_h - mu_h)**2)/(2*sigma_h**2))/(sigma_h * np.sqrt(2*np.pi))
    
    y_w /= np.max(y_w)
    y_h /= np.max(y_h)
    
    y_w = np.matrix(y_w.reshape(1,len(y_w)))
    y_h = np.matrix(y_h.reshape(len(y_h),1))
    
#    plt.plot(np.array(y_w).reshape(-1))
    
    gauss_map = y_h * y_w
    gauss_map = np.array(gauss_map)
    gauss_map /= np.max(gauss_map)
    
    mask = np.zeros([img_h,img_w])
    mask[int(cy-h*1.1/2):int(cy+h*1.1/2),
         int(cx-w*1.1/2):int(cx+w*1.1/2)] = 1
#    gauss_map = gauss_map*mask
    
    return gauss_map
    

def gauss_map1(cx,cy,sz_w,sz_h,map_w,map_h,output_sigma_factor):
            
    x_w = np.linspace(0,map_w-1,map_w)
    x_h = np.linspace(0,map_h-1,map_h)
    
    x_w = np.repeat(x_w[None,:],map_h,0) - cx
    x_h = np.repeat(x_h[:,None],map_w,1) - cy
    
    output_sigma = np.sqrt(sz_w*sz_h) * output_sigma_factor;
    gauss_map = np.exp(-0.5 * (((x_w**2 + x_h**2) / output_sigma**2)));
    
    return gauss_map

def gauss_label( target_sz, feat_sz):
        
    sigma = 0.1 * np.ceil(target_sz/4.)
    
    if np.max(target_sz)/np.min(target_sz) < 1.025 and np.max(target_sz)>120:
        alpha = 0.2
    else:
        alpha = 0.3


    mu_w = (feat_sz-1)/2
    mu_h = (feat_sz-1)/2
        

    x_w = np.linspace(0,feat_sz-1,feat_sz)
    x_h = np.linspace(0,feat_sz-1,feat_sz)
    
    y_w = np.exp(- alpha*( (x_w - mu_w)**2 / sigma[1]**2 ) )
    y_h = np.exp(- alpha*( (x_h - mu_h)**2 / sigma[0]**2 ) )
    
    y_w /= np.max(y_w)
    y_h /= np.max(y_h)
    
    y_w = np.matrix(y_w.reshape(1,len(y_w)))
    y_h = np.matrix(y_h.reshape(len(y_h),1))
    
    
    gauss_map = y_h * y_w
    gauss_map = np.array(gauss_map)
    gauss_map /= np.max(gauss_map)
   
    
    return gauss_map

# %%
def trans_bbox(bbox, type):
    if type == 'center':
        bbox[0] = bbox[0]+bbox[2]/2
        bbox[1] = bbox[1]+bbox[3]/2
        return bbox
    elif type == 'xy':
        bbox[0] = bbox[0]-bbox[2]/2
        bbox[1] = bbox[1]-bbox[3]/2
        return bbox
        

def process_ground_truth(bbox, type='left-top'):
    """
    Process ground truth box.
    """
    if type == 'center':
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
            
            x = x + w/2
            y = y + h/2
        # [x,y,w,h]
        elif bbox.shape[1] == 4:
            x = bbox[:,0::4]
            y = bbox[:,1::4]
            w = bbox[:,2::4]
            h = bbox[:,3::4]

            x = x + w/2
            y = y + h/2
            
        return np.hstack((x,y,w,h))
        
    else:
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




def overlap_ratio(bbox, boxes):
    """
    Compute overlap ratio.
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2] -1
    y2 = bbox[1] + bbox[3] -1
    
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

#%%

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

    
def generate_samples(img, gtbox, n, trans_f=0.3, scale_f=1.5, aspect_f=1.1, display=False):
    """
    Generate samples.
    """
    bbox = gtbox.copy()
    img_size = np.array(img.shape[:2][::-1])

    # (center_x, center_y, w, h)
    sample = np.array([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]], dtype='float32')
    samples = np.tile(sample[None,:],(n,1))

    # vary aspect ratio
    ratio = np.random.rand(n,1)*2-1
    samples[:,2:] *= aspect_f ** np.concatenate([ratio, -ratio],axis=1)

    # sample generation
    samples[:,:2] += trans_f * np.mean(bbox[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
    samples[:,2:] *= scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)


    # adjust bboxox range
    samples[:,2:] = np.clip(samples[:,2:], 10, img_size-10)
    samples[:,:2] = np.clip(samples[:,:2], 0, img_size)

    # (min_x, min_y, w, h)
    samples[:,:2] -= samples[:,2:]/2

    r = overlap_ratio(bbox, samples)
    s = np.prod(samples[:,2:], axis=1) / np.prod(bbox[2:])
    idx = (r >= 0.6) * (r <= 1) * \
          (s >= 1) * (s <= 2)

    samples = samples[idx]




    samples = samples.astype(int)
    
    
    

    """display boxes"""
    if display:
        im_copy = img.copy()
        for i in range(len(samples)):
            cv2.rectangle(im_copy,
                          (samples[i,0],samples[i,1]),
                          (samples[i,0]+samples[i,2], samples[i,1]+samples[i,3]),
                          (0,255,0), 1)
        plt.imshow(im_copy)

    return samples
    
#%%
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



def crop_to_bigger(img, box, times, off):
    """
    Enlarge bbox and crop img according to the bbox.
    """
    im = img.copy()
    bbox = box.copy()
    
    # enlarge bbox 3 times
    cx = bbox[0]+bbox[2]/2
    cy = bbox[1]+bbox[3]/2
    longer = max(bbox[2],bbox[3])
    
    if len(img.shape)>2:
#        ch0 = np.lib.pad(im[:,:,0],(int(longer*times),int(longer*times)),'edge')
#        ch1 = np.lib.pad(im[:,:,1],(int(longer*times),int(longer*times)),'edge')
#        ch2 = np.lib.pad(im[:,:,2],(int(longer*times),int(longer*times)),'edge')
        ch0 = np.lib.pad(im[:,:,0],(int(longer*times),int(longer*times)),'constant', constant_values=103.939)
        ch1 = np.lib.pad(im[:,:,1],(int(longer*times),int(longer*times)),'constant', constant_values=116.779)
        ch2 = np.lib.pad(im[:,:,2],(int(longer*times),int(longer*times)),'constant', constant_values=123.68)
        im = np.dstack((ch0,ch1,ch2))
    else:
        im = np.lib.pad(im[:,:],(int(longer*times),int(longer*times)),'edge')
    cx += int(longer*times)
    cy += int(longer*times)
        
    
    bbox[0] = int(cx -bbox[2]*times/2.)
    bbox[1] = int(cy -bbox[3]*times/2.)


        
    crop = crop_and_pad(im,bbox[0], bbox[1],
                            bbox[0]+int(bbox[2]*times),
                            bbox[1]+int(bbox[3]*times))    

    return crop    



def crop_to_bigger2(img, box, times):
    """
    Do not return new bbox.
    """
    im = img.copy()
    bbox = box.copy()
    
    # enlarge bbox 3 times
    cx = bbox[0]+bbox[2]/2
    cy = bbox[1]+bbox[3]/2
    
    l = abs(min(0,cx - np.floor(bbox[2]*times/2)))
    r = abs(im.shape[1]-max(im.shape[1],cx + np.ceil(bbox[2]*times/2)))
    u = abs(min(0,cy - np.floor(bbox[3]*times/2)))
    d = abs(im.shape[0]-max(im.shape[0],cy + np.ceil(bbox[3]*times/2)))
    
    if len(im.shape)>2:
        ch0 = np.lib.pad(im[:,:,0],((int(u),int(d)),(int(l),int(r))),'constant', constant_values=np.mean(im[:,:,0]))
        ch1 = np.lib.pad(im[:,:,1],((int(u),int(d)),(int(l),int(r))),'constant', constant_values=np.mean(im[:,:,1]))
        ch2 = np.lib.pad(im[:,:,2],((int(u),int(d)),(int(l),int(r))),'constant', constant_values=np.mean(im[:,:,2]))
        im = np.dstack((ch0,ch1,ch2))
    else:
        im = np.lib.pad(im[:,:],(((int(u),int(d)),(int(l),int(r)))),'edge')

    cx += l
    cy += u

    x1 = int(cx - np.floor(bbox[2]*times/2))       
    y1 = int(cy - np.floor(bbox[3]*times/2))      
    x2 = int(cx + np.ceil(bbox[2]*times/2))    
    y2 = int(cy + np.ceil(bbox[3]*times/2))    

    bbox[0] = cx - bbox[2]/2 -x1
    bbox[1] = cy - bbox[3]/2 -y1

    return im[y1:y2,x1:x2],bbox
    
def crop_to_bigger3(img, box, times):
    """
    Do not change ratio.
    """
    im = img.copy()
    bbox = box.copy()
    
    # enlarge bbox 3 times
    cx = bbox[0]+bbox[2]/2
    cy = bbox[1]+bbox[3]/2
    longer = max(bbox[2],bbox[3])
    
    l = abs(min(0,cx - np.floor(longer*times/2)))
    r = abs(im.shape[1]-max(im.shape[1],cx + np.ceil(longer*times/2)))
    u = abs(min(0,cy - np.floor(longer*times/2)))
    d = abs(im.shape[0]-max(im.shape[0],cy + np.ceil(longer*times/2)))
    
    if len(im.shape)>2:
        ch0 = np.lib.pad(im[:,:,0],((int(u),int(d)),(int(l),int(r))),'constant', constant_values=np.mean(im[:,:,0]))
        ch1 = np.lib.pad(im[:,:,1],((int(u),int(d)),(int(l),int(r))),'constant', constant_values=np.mean(im[:,:,1]))
        ch2 = np.lib.pad(im[:,:,2],((int(u),int(d)),(int(l),int(r))),'constant', constant_values=np.mean(im[:,:,2]))
        im = np.dstack((ch0,ch1,ch2))
    else:
        im = np.lib.pad(im[:,:],(((int(u),int(d)),(int(l),int(r)))),'edge')

    cx += l
    cy += u

    x1 = int(cx - np.floor(longer*times/2))       
    y1 = int(cy - np.floor(longer*times/2))      
    x2 = int(cx + np.ceil(longer*times/2))    
    y2 = int(cy + np.ceil(longer*times/2))    

    bbox[0] = cx - bbox[2]/2 -x1
    bbox[1] = cy - bbox[3]/2 -y1

    return im[y1:y2,x1:x2],bbox
    

def crop_and_pad(img, x1,y1,x2,y2):
    """
    Crop img patch and pad with mean.
    """
    
    img_w = img.shape[1]
    img_h = img.shape[0]

    # compute mean
    if len(img.shape)>2:
        mean = np.array( [np.mean(img[:,:,0]),
                          np.mean(img[:,:,1]),
                          np.mean(img[:,:,2])], dtype=float )
    else:
        mean = np.array( [114.80], dtype=float )
        
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
        
    crop = img[y1:y2,x1:x2]
    
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
    
    return crop

#%%    

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
        seq_list.remove('Jogging')
        seq_list.append('Jogging-1')
        seq_list.append('Jogging-2')
        seq_list.remove('Skating2')
        seq_list.append('Skating2-1')
        seq_list.append('Skating2-2')
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


    elif vid_set == 'vot2018':
        # read ground truth of first video
        with open(vid_dir + '/' + vid_list[vid_id] + '/groundtruth.txt', 'r') as gt_file:
            gt_box = gt_file.readlines()
            frame_num = len(gt_box)
        
        # remove '\n' and convert str list to float array
        for index, gt_box_i in enumerate(gt_box):
            gt_box[index] = np.array(map( float,gt_box_i.splitlines()[0].split(',') ), dtype=int)   
        
        # read first frame (B,G,R)
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/color/{:0>8d}.jpg'.format(1))
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
        
        return frame,np.array(gt_box),frame_num
        
    elif vid_set == 'otb':
        # read ground truth of first video
        if vid_list[vid_id] == 'Tiger1':
            with open(vid_dir + '/' + vid_list[vid_id] + '/groundtruth_rect.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()[5:]
                frame_num = len(gt_box)
                               
        elif vid_list[vid_id] == 'Jogging-1':
            with open(vid_dir + '/Jogging' + '/groundtruth_rect.1.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)
                
        elif vid_list[vid_id] == 'Jogging-2':
            with open(vid_dir + '/Jogging' + '/groundtruth_rect.2.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)
                
        elif vid_list[vid_id] == 'Skating2-1':
            with open(vid_dir + '/Skating2' + '/groundtruth_rect.1.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)
                
        elif vid_list[vid_id] == 'Skating2-2':
            with open(vid_dir + '/Skating2' + '/groundtruth_rect.2.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)

        elif vid_list[vid_id] == 'Human4':
            with open(vid_dir + '/Human4' + '/groundtruth_rect.2.txt', 'r') as gt_file:
                gt_box = gt_file.readlines()
                frame_num = len(gt_box)

        else:
            with open(vid_dir + '/' + vid_list[vid_id] + '/groundtruth_rect.txt', 'r') as gt_file:
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
                    
                    
                    
        if vid_list[vid_id] == 'Jogging-1' or vid_list[vid_id] == 'Jogging-2':
            for (dirpath, seq_list, filenames) in walk(vid_dir + '/Jogging' + '/img/'):
                break
            filenames.sort()
            frame = cv2.imread(vid_dir + '/Jogging' + '/img/' + filenames[0])
            
        elif vid_list[vid_id] == 'Skating2-1' or vid_list[vid_id] == 'Skating2-2':
            for (dirpath, seq_list, filenames) in walk(vid_dir + '/Skating2' + '/img/'):
                break
            filenames.sort()
            frame = cv2.imread(vid_dir + '/Skating2' + '/img/' + filenames[0])
            
        else:           
            # get img list, reason for clutter img name
            for (dirpath, seq_list, filenames) in walk(vid_dir + '/' + vid_list[vid_id] + '/img/'):
                break
            filenames.sort()
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
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/{:0>8d}.jpg'.format(frame_id))
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
        
        return frame

    elif vid_set == 'vot2018':
        frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/color/{:0>8d}.jpg'.format(frame_id))
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
        
        return frame
        
        
    elif vid_set == 'otb':
        if vid_list[vid_id] == 'Jogging-1' or vid_list[vid_id] == 'Jogging-2':
            for (dirpath, seq_list, filenames) in walk(vid_dir + '/Jogging' + '/img/'):
                break
            filenames.sort()
            frame = cv2.imread(vid_dir + '/Jogging' + '/img/' + filenames[frame_id-1])
            
        elif vid_list[vid_id] == 'Skating2-1' or vid_list[vid_id] == 'Skating2-2':
            for (dirpath, seq_list, filenames) in walk(vid_dir + '/Skating2' + '/img/'):
                break
            filenames.sort()
            frame = cv2.imread(vid_dir + '/Skating2' + '/img/' + filenames[frame_id-1])
            
        else:           
            # get img list, reason for clutter img name
            for (dirpath, seq_list, filenames) in walk(vid_dir + '/' + vid_list[vid_id] + '/img/'):
                break
            filenames.sort()
        
            if vid_list[vid_id] == 'Tiger1':
                frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1 +5])
            elif vid_list[vid_id] == 'David':
                frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1 +299])
#            elif vid_list[vid_id] == 'BlurCar1':
#                frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1 +246])
#            elif vid_list[vid_id] == 'BlurCar3':
#                frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1 +2])
#            elif vid_list[vid_id] == 'BlurCar4':
#                frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1 +17])

            else:
                frame = cv2.imread(vid_dir + '/' + vid_list[vid_id] + '/img/' + filenames[frame_id-1])
        
        if len(frame.shape) == 2:
            frame = np.dstack((frame,frame,frame))
    
        return frame

# %%
    
if __name__ == "__main__":

    print("This is a module for tracker.")
