#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Function for RPN Tracker.


Created on Thu Oct 19 08:31:04 2017

@author: zhaohj
  
"""



#%%
import sys

sys.path.append('/home/zhaohj/Documents/Workspace/TDT_VGG_py/lib')
import tools
import rpn

import cv2
import numpy as np
import matplotlib.pylab as plt
import random
import math

def xy_to_input(box,fw,fh,origin):
    bbox = box.copy().astype(float)
    bbox[0] = int(bbox[0] - origin[0])
    bbox[1] = int(bbox[1] - origin[1])
    bbox[0:4:2] = bbox[0:4:2]*fw
    bbox[1:4:2] = bbox[1:4:2]*fh
    bbox = bbox.astype(int)
    return bbox


def xy_to_img(box,fw,fh,origin):
    bbox = box.copy().astype(float)
    bbox[0:4:2] = bbox[0:4:2]/fw
    bbox[1:4:2] = bbox[1:4:2]/fh
    bbox[0] = int(bbox[0] + origin[0])
    bbox[1] = int(bbox[1] + origin[1])
    bbox = bbox.astype(int)
    return bbox

def scale(predd,t):
    pred = predd.copy()
    cx = np.average(np.where(pred > 0.8)[1]).astype(int)
    cy = np.average(np.where(pred > 0.8)[0]).astype(int)
    pred[pred>0.1] = 1

    
    for f in range(1,50):
        f2 = int(min(cy+(f+1)*2, 223))
        f1 = int(min(cy+f*2, 223))
        up = (np.sum(pred[cy:f2, cx-1:cx+1]) - 
              np.sum(pred[cy:f1, cx-1:cx+1]) )\
              /np.sum(pred[cy:f1, cx-1:cx+1])
        if up < t:
            break
    h1 = f*2
    h1 = min(cy+h1, 223)-cy
        
    for f in range(1,50):
        f2 = int(max(cy-(f+1)*2, 0))
        f1 = int(max(cy-f*2, 0))
        up = (np.sum(pred[f2:cy, cx-1:cx+1]) - 
              np.sum(pred[f1:cy, cx-1:cx+1]) )\
              /np.sum(pred[f1:cy, cx-1:cx+1])
        if up < t:
            break
    h2 = f*2
    h2 = cy-max(cy-h2, 0)
    
    h = h1 + h2

    for f in range(1,50):
        f2 = int(min(cx+(f+1)*2, 223))
        f1 = int(min(cx+f*2, 223))
        up = (np.sum(pred[cy-1:cy+1, cx:f2]) - 
              np.sum(pred[cy-1:cy+1, cx:f1]) )\
              /np.sum(pred[cy-1:cy+1, cx:f1])
        if up < t:
            break
    w1 = f*2
    w1 = min(cx+w1, 223)-cx
    
    for f in range(1,50):
        f2 = int(max(cx-(f+1)*2, 0))
        f1 = int(max(cx-f*2, 0))
        up = (np.sum(pred[cy-1:cy+1, f2:cx]) - 
              np.sum(pred[cy-1:cy+1, f1:cx]) )\
              /np.sum(pred[cy-1:cy+1, f1:cx])
        if up < t:
            break
    w2 = f*2
    w2 = cx-max(cx-w2, 0)
    
    w = w1 + w2
        
    return np.array([cx-w2,cy-h2,w,h]).astype(int)
    

def gauss_map(cx,cy,w,h,img_w,img_h):

    if w == 0 or h == 0:
        return np.zeros([img_h,img_w])
    
    mu_w = cx
    mu_h = cy
    
    # [-n sigma, +n sigma]
    sigma_w = w/4.0
    sigma_h = h/4.0
    
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
    
    

def histogram(pred, up):
    s,d = np.histogram(pred)
        
    summ = 0
    for i in range(10):
        summ += s[i]
        if summ/(224*224.0) > up:
            return i+1

    
    
def find_max_contour(input_bin):

    _,contours,hierarchy = cv2.findContours(input_bin.astype('uint8'),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE) 
    area = []
    for c in contours:
        area.append(cv2.contourArea(c))
        
    if len(area) != 0:
        return contours[np.argmax(np.array(area))],len(area)
    else:
        return np.array([]),0


def gen_anchors():
    
    basic_anchor = np.array([[-28 ,-56 ,27 ,55 ],
                             [-56 ,-56 ,55 ,55 ],
                             [-56 ,-28 ,55 ,27 ]],dtype='float64')
    anchors = rpn.generate_anchors(basic_anchor, 16, (14,14))
    del basic_anchor
    
    # remove anchor boxes which out of the input img
    for i in range(len(anchors)):
        if anchors[i,0] < 0 or anchors[i,1] < 0 \
            or anchors[i,2] > 223 or anchors[i,3] > 223:
            anchors[i,:] = np.array([0,0,0,0])
    
    # to [x,y,w,h]
    # -1 to make the w/h of invalid anchors =1
    anchors[:,2::4] -= anchors[:,0::4] - 1
    anchors[:,3::4] -= anchors[:,1::4] - 1
    
    return anchors.astype(int)

    
def img_to_test(img, bbox,dis):
    
    bbox_test = bbox.copy()
    
    # center of bbox
    cx = int(bbox_test[0] + bbox_test[2]/2)
    cy = int(bbox_test[1] + bbox_test[3]/2)
    
    f = 2
    longer = max(bbox_test[2], bbox_test[3])
    side_w = int(longer*f)
    side_h = int(longer*f)
    
    temp_x1 = int(cx - side_w/2)
    temp_y1 = int(cy - side_h/2)
    temp_x2 = int(cx + side_w/2)+1
    temp_y2 = int(cy + side_h/2)+1
    point_lu = np.array([temp_x1,temp_y1]).astype(int)
    
        
    # subtract left-up point of crop box
    img_crop = tools.crop_and_pad(img,temp_x1,temp_y1,temp_x2,temp_y2)
    
    bbox_crop = np.zeros(4,dtype=int)
    bbox_crop[0] = bbox_test[0] - temp_x1
    bbox_crop[1] = bbox_test[1] - temp_y1
    bbox_crop[2] = bbox_test[2]
    bbox_crop[3] = bbox_test[3]

    img_crop = img_crop.astype('uint8')
    bbox_crop = bbox_crop.astype(int)
    
    # plot bbox
    if dis:
        plt.figure(10)
        plt.imshow(cv2.rectangle(img_crop.copy(),
                                 (bbox_crop[0],bbox_crop[1]),
                                 (bbox_crop[0]+bbox_crop[2],
                                  bbox_crop[1]+bbox_crop[3]),
                                  (0,0,255), 1)[:,:,::-1])
                             
    return img_crop, bbox_crop, point_lu


    

def img_to_sample(img, bbox, flip,bright,patch, dis):  
    
    bbox_sample = bbox.copy()
    
    # center of bbox
    cx = int(bbox_sample[0] + bbox_sample[2]/2)
    cy = int(bbox_sample[1] + bbox_sample[3]/2)

    # crop scale
    f = random.uniform( 1.5, 2.5 )
#    f = random.uniform( 1.3, 1.8 )
    
    longer = max(bbox_sample[2], bbox_sample[3])
    side = int(longer*f)
    
    # crop box
    rand_x = random.uniform(-int(side/2.-bbox_sample[2]/2.),
                             int(side/2.-bbox_sample[2]/2.))
    rand_y = random.uniform(-int(side/2.-bbox_sample[3]/2.),
                             int(side/2.-bbox_sample[3]/2.))

    
    crop_x1 = int(cx + rand_x - side/2)
    crop_y1 = int(cy + rand_y - side/2)
    crop_x2 = int(cx + rand_x + side/2)
    crop_y2 = int(cy + rand_y + side/2)
    point_lu = np.array([crop_x1, crop_y1]).astype(int)
    # to [x,y,w,h]
    temp_bbox = np.zeros(4, dtype=int)
    temp_bbox = np.array([crop_x1, crop_y1, crop_x2-crop_x1, crop_y2-crop_y1])    
    
    samples = temp_bbox.copy().reshape(1,4)

    
    # crop sample img
    img_sample = tools.crop_and_pad(img,
                                    samples[0,0],samples[0,1],
                                    samples[0,0]+samples[0,2],
                                    samples[0,1]+samples[0,3])
     
    # to x,y,x,y
    bbox_sample[2] = bbox_sample[2] + bbox_sample[0]
    bbox_sample[3] = bbox_sample[3] + bbox_sample[1]
    # subtract left-up point of crop box
    bbox_sample[0] = max(0, min(bbox_sample[0] - samples[0,0], side-1))
    bbox_sample[1] = max(0, min(bbox_sample[1] - samples[0,1], side-1))
    bbox_sample[2] = max(0, min(bbox_sample[2] - samples[0,0], side-1))
    bbox_sample[3] = max(0, min(bbox_sample[3] - samples[0,1], side-1))
    # to x,y,w,h
    bbox_sample[2] = bbox_sample[2] - bbox_sample[0] -1
    bbox_sample[3] = bbox_sample[3] - bbox_sample[1] -1
    
                             
    # Gaussian Blur
    if math.floor(random.uniform(0,2)):
        img_sample = cv2.GaussianBlur(img_sample.astype(float), (5,5), 2)

    # random flip
    if flip:
        if math.floor(random.uniform(0,2)):
            img_sample = cv2.flip(img_sample, 1)   
            bbox_sample[1] = bbox_sample[1]
            bbox_sample[0] = img_sample.shape[1]-bbox_sample[0]-bbox_sample[2]


    # random change brightness
    if bright:
        img_temp = img_sample.copy().astype(int)
        
        img_temp += int(random.uniform( -20, 100 ))   
    
        # brightness patch
        if math.floor(random.uniform(0,9)):
            temp_mask = np.zeros([ samples[0,3],samples[0,2] ])
            temp_mask[bbox_sample[1]:bbox_sample[1]+bbox_sample[3],
                      bbox_sample[0]:bbox_sample[0]+bbox_sample[2]]\
            = random.uniform( -20, 20 )
            
            cx = random.uniform(bbox_sample[0],bbox_sample[0]+bbox_sample[2])
            cy = random.uniform(bbox_sample[1],bbox_sample[1]+bbox_sample[3])
            gaus = gauss_map(cx,cy,bbox_sample[2],bbox_sample[3],
                             samples[0,2],samples[0,3]).copy()
            temp_mask *= gaus
            
            img_temp += np.dstack((temp_mask,temp_mask,temp_mask)).astype(int) 

            
            
                                      
        # limit bright"""
        img_temp[img_temp > 255] = 255
        img_temp[img_temp < 0] = 0
        img_sample = img_temp

    
    # random patch
    if patch:
        if math.floor(random.uniform(0,9)):
    
            mean = np.dstack((np.mean(img_sample[:,:,0]),
                              np.mean(img_sample[:,:,1]),
                              np.mean(img_sample[:,:,2])) ).astype(int)
            
            off_y = int(random.uniform(bbox_sample[1]+bbox_sample[3]*0.0,
                                       bbox_sample[1]+bbox_sample[3]*1.0))   
            off_x = int(random.uniform(bbox_sample[0]+bbox_sample[2]*0.0,
                                       bbox_sample[0]+bbox_sample[2]*1.0))
            
            
            off_w = int((bbox_sample[2]+bbox_sample[3])*random.uniform(0.1,0.2))
            off_h = int((bbox_sample[2]+bbox_sample[3])*random.uniform(0.1,0.2))
            
            img_sample[off_y-off_h:off_y+off_h,
                       off_x-off_w:off_x+off_w,:]=mean
    
    img_sample = img_sample.astype('uint8')
    bbox_sample = bbox_sample.astype(int)

    
    # plot bbox in the sample img
    if dis:
        plt.figure(10)
        plt.imshow(cv2.rectangle(img_sample.copy(),
                                 (bbox_sample[0],bbox_sample[1]),
                                 (bbox_sample[0]+bbox_sample[2],
                                  bbox_sample[1]+bbox_sample[3]),
                                  (0,0,255), 1)[:,:,::-1])
                                 
    return img_sample, bbox_sample, point_lu


def img_to_sample1(img, bbox,f_list, dis):  
    
    bbox_sample = bbox.copy().astype(float)
    
    # center of bbox
    cx = int(bbox_sample[0] + bbox_sample[2]/2)
    cy = int(bbox_sample[1] + bbox_sample[3]/2)
    longer = max(bbox_sample[2], bbox_sample[3])
        
    img_buf = []
    samples_buf = []
    bbox_buf = np.zeros([1,4])

    # crop scale
    for f in f_list:
    
        side = int(longer*f)
        
        # crop box
        off_x = (side-bbox[2])/2.
        off_y = (side-bbox[3])/2.
        
        off = np.array([
                        [     0,     0,     0,     0],
                        [ off_x,     0, off_x,     0],
                        [-off_x,     0,-off_x,     0],
                        [     0, off_y,     0, off_y],
                        [     0,-off_y,     0,-off_y],
                        [ off_x/2,     0, off_x/2,     0],
                        [-off_x/2,     0,-off_x/2,     0],
                        [     0, off_y/2,     0, off_y/2],
                        [     0,-off_y/2,     0,-off_y/2],
                        [ off_x/4,     0, off_x/4,     0],
                        [-off_x/4,     0,-off_x/4,     0],
                        [     0, off_y/4,     0, off_y/4],
                        [     0,-off_y/4,     0,-off_y/4]
                        ]).astype(int)
                                
        # x1, y1, x2, y2
        crop = np.array([cx - side/2, cy - side/2,
                         cx + side/2, cy + side/2]).reshape(1,4)
        crop = np.repeat(crop, 13, 0)
        crop = crop + off
        bbox_sample = np.repeat(bbox.reshape(1,4), 13,0)

        # to [x,y,w,h]
        crop[:,2:] = crop[:,2:] - crop[:,:2]  
        for i in range(len(crop)):  
            # crop sample img
            img_sample = tools.crop_and_pad(img,
                                            int(crop[i,0]),int(crop[i,1]),
                                            int(crop[i,0]+crop[i,2]),
                                            int(crop[i,1]+crop[i,3]))
        
            # bbox to x,y,x,y
            bbox_sample[i,2] = bbox_sample[i,2] + bbox_sample[i,0]
            bbox_sample[i,3] = bbox_sample[i,3] + bbox_sample[i,1]
            # subtract left-top point of crop box
            bbox_sample[i,0] = max(0, min(bbox_sample[i,0] - crop[i,0], side-1))
            bbox_sample[i,1] = max(0, min(bbox_sample[i,1] - crop[i,1], side-1))
            bbox_sample[i,2] = max(0, min(bbox_sample[i,2] - crop[i,0], side-1))
            bbox_sample[i,3] = max(0, min(bbox_sample[i,3] - crop[i,1], side-1))
            # to x,y,w,h
            bbox_sample[i,2] = bbox_sample[i,2] - bbox_sample[i,0] -1
            bbox_sample[i,3] = bbox_sample[i,3] - bbox_sample[i,1] -1


            fw = img_sample.shape[1]/224.0
            fh = img_sample.shape[0]/224.0
            bbox_sample[i,::2] = bbox_sample[i,::2]/fw
            bbox_sample[i,1::2] = bbox_sample[i,1::2]/fh

            img_sample = cv2.resize(img_sample,(224,224),cv2.INTER_LINEAR)
            
        
            img_buf.append(img_sample)
            samples_buf.append(crop[i].reshape(-1))
        
        bbox_buf = np.vstack((bbox_buf,bbox_sample))
    
    bbox_buf = bbox_buf[1:,:].astype(int)                             

    samples_pos = np.array(samples_buf)
    
    # plot bbox in the sample img
    if dis:
        for i in range(len(bbox_buf)):
            plt.figure(i+1)
            plt.imshow(cv2.rectangle(img_buf[i].copy(),
                                     (bbox_buf[i,0],bbox_buf[i,1]),
                                     (bbox_buf[i,0]+bbox_buf[i,2],
                                      bbox_buf[i,1]+bbox_buf[i,3]),
                                      (0,0,255), 1)[:,:,::-1])
#        im_copy = img.copy()
#        for i in range(len(samples_pos)):
#            cv2.rectangle(im_copy,
#                          (samples_pos[i,0],samples_pos[i,1]),
#                          (samples_pos[i,0]+samples_pos[i,2],
#                           samples_pos[i,1]+samples_pos[i,3]),
#                          (0,255,0), 1)
#        plt.imshow(im_copy)
                            
    return samples_pos, bbox_buf
    

def img_to_sample2(img, bbox, dis):  
 
    bbox_sample = bbox.copy().astype(float)
    
    # center of bbox
    cx = int(bbox_sample[0] + bbox_sample[2]/2)
    cy = int(bbox_sample[1] + bbox_sample[3]/2)
    longer = max(bbox_sample[2], bbox_sample[3])
    
    bbox_temp = np.array([cx-longer/2,cy-longer/2,longer,longer],dtype=int)
    samples_neg = tools.generate_samples(img, bbox_temp, 2000, 'whole',
                                         0, 0, -1, 0.2, 0, dis)
    
    
    if dis:
        im_copy = img.copy()
        for i in range(len(samples_neg)):
            cv2.rectangle(im_copy,
                          (samples_neg[i,0],samples_neg[i,1]),
                          (samples_neg[i,0]+samples_neg[i,2],
                           samples_neg[i,1]+samples_neg[i,3]),
                          (0,255,0), 1)
        plt.imshow(im_copy)
        plt.show()

    return samples_neg

    

def img_to_input(img, bbox, dis):
    
    bbox_temp = bbox.copy()
    
    # resize img to 241*241
    factor_w = 224.0/img.shape[1]
    factor_h = 224.0/img.shape[0]
    img_input = cv2.resize(img.copy().astype(float),
                           None,None,factor_w,factor_h).astype('uint8')
    
    # bbox resize to input
    bbox_input = np.zeros(4,dtype=int)
    bbox_input[0:4:2] = bbox_temp[0:4:2]*factor_w
    bbox_input[1:4:2] = bbox_temp[1:4:2]*factor_h
    bbox_input = bbox_input.astype(int)
    
    # plot bbox
    if dis:
        plt.imshow(cv2.rectangle(img_input.copy(),
                                 (bbox_input[0],bbox_input[1]),
                                 (bbox_input[0]+bbox_input[2],
                                  bbox_input[1]+bbox_input[3]),
                                  (0,0,255), 1)[:,:,::-1])                        
    return img_input, bbox_input, factor_w,factor_h
    

def gen_cls_labels(img_input, bbox, anchors, iouu, dis):
    
    bbox_input = bbox.copy()
    
    # compute iou
    iou = tools.overlap_ratio(bbox_input,anchors.reshape(-1,4))
    iou = iou/np.max(iou)
    
    # set invalid labels
    labels = np.zeros(anchors.shape[0],dtype=int)
    labels[anchors[:,2]==1] = 0
    
    # set pos
    labels[iou>iouu] = 1
    labels[iou<0.3] = 0
    
#    # ignore some samples to keep sample balance
#    pos_num = np.sum(cls_label[cls_label==1])
#    
#    temp_where = np.where(labels == 0)[0]
#    if temp_where.size > 64:
#        temp_ignore = np.random.choice(temp_where,
#                                       size=(len(temp_where)-32),replace=False)
#        labels[temp_ignore] = -1
#
#    temp_where = np.where(labels == 1)[0]
#    if temp_where.size > 64:
#        temp_ignore = np.random.choice(temp_where,
#                                       size=(len(temp_where)-32),replace=False)
#        labels[temp_ignore] = -1
    
    # plot iou
    if dis:
        temp_iou = iou.reshape(14,14,3).transpose(2,0,1)
        temp_labels = labels.reshape(14,14,3).transpose(2,0,1)
        
        fig = plt.figure(10)
        

        fig.add_subplot(3,3,1)
        plt.imshow(cv2.rectangle(img_input.astype('uint8'),
                                 (bbox_input[0],bbox_input[1]),
                                 (bbox_input[0]+bbox_input[2],
                                  bbox_input[1]+bbox_input[3]),
                                  (0,0,255), 1)[:,:,::-1])
        for i in range(3):
            fig.add_subplot(3,3,i+4)
            plt.imshow(temp_iou[i],cmap='Reds')
            
        for i in range(3):
            fig.add_subplot(3,3,i+7)
            plt.imshow(temp_labels[i],interpolation='nearest')
            
        fig.tight_layout()
        
    return labels.reshape(14,14,3).transpose(2,0,1)    


def gen_skip_labels(img_input,bbox,gauss,dis):

    bbox_input = bbox.copy()  
                
    if bbox_input[3] == 0:
        # back labels
        skip1_label = np.zeros([224,224])
        
        skip2_label = np.zeros([112,112])
        
        skip3_label = np.zeros([56,56])
        
        skip4_label = np.zeros([28,28])
        
        skip5_label = np.zeros([14,14])

        return skip1_label, skip2_label, skip3_label, skip4_label, skip5_label
        
    # back labels
    labels_input = np.zeros(img_input.shape[0:2])
    labels_input[bbox_input[1]:(bbox_input[1]+bbox_input[3]),
                 bbox_input[0]:(bbox_input[0]+bbox_input[2])] = 1
        

    skip1_label = cv2.resize(labels_input,(224,224),
                      interpolation=cv2.INTER_NEAREST).astype(float)
    
    skip2_label = cv2.resize(labels_input,(112,112),
                      interpolation=cv2.INTER_NEAREST).astype(float)
    
    skip3_label = cv2.resize(labels_input,(56,56),
                      interpolation=cv2.INTER_NEAREST).astype(float)
    
    skip4_label = cv2.resize(labels_input,(28,28),
                      interpolation=cv2.INTER_NEAREST).astype(float)
    
    skip5_label = cv2.resize(labels_input,(14,14),
                      interpolation=cv2.INTER_NEAREST).astype(float)
    
    
    if gauss:
        skip1_label = gauss_map(int(bbox_input[0]+bbox_input[2]/2.0),
                                int(bbox_input[1]+bbox_input[3]/2.0),
                                bbox_input[2],bbox_input[3],224,224)
#        skip1_label[:bbox_input[1],:] = 0
#        skip1_label[bbox_input[1]+bbox_input[3]:,:] = 0
#        skip1_label[:,:bbox_input[0]] = 0
#        skip1_label[:,bbox_input[0]+bbox_input[2]:] = 0

        skip2_label = gauss_map(int((bbox_input[0]+bbox_input[2]/2.0)/2.0),
                                int((bbox_input[1]+bbox_input[3]/2.0)/2.0),
                                int(bbox_input[2]/2.0),
                                int(bbox_input[3]/2.0),112,112)
#        skip2_label[:int(bbox_input[1]/2.0),:] = 0
#        skip2_label[int((bbox_input[1]+bbox_input[3])/2.0):,:] = 0
#        skip2_label[:,:int(bbox_input[0]/2.0)] = 0
#        skip2_label[:,int((bbox_input[0]+bbox_input[2])/2.0):] = 0

        skip3_label = gauss_map(int((bbox_input[0]+bbox_input[2]/2.0)/4.0),
                                int((bbox_input[1]+bbox_input[3]/2.0)/4.0),
                                int(bbox_input[2]/4.0),
                                int(bbox_input[3]/4.0),56,56)
#        skip3_label[:int(bbox_input[1]/4.0),:] = 0
#        skip3_label[int((bbox_input[1]+bbox_input[3])/4.0):,:] = 0
#        skip3_label[:,:int(bbox_input[0]/4.0)] = 0
#        skip3_label[:,int((bbox_input[0]+bbox_input[2])/4.0):] = 0

        skip4_label = gauss_map(int((bbox_input[0]+bbox_input[2]/2.0)/8.0),
                                int((bbox_input[1]+bbox_input[3]/2.0)/8.0),
                                int(bbox_input[2]/8.0),
                                int(bbox_input[3]/8.0),28,28)
#        skip4_label[:int(bbox_input[1]/8.0),:] = 0
#        skip4_label[int((bbox_input[1]+bbox_input[3])/8.0):,:] = 0
#        skip4_label[:,:int(bbox_input[0]/8.0)] = 0
#        skip4_label[:,int((bbox_input[0]+bbox_input[2])/8.0):] = 0

        skip5_label = gauss_map(int((bbox_input[0]+bbox_input[2]/2.0)/16.0),
                                int((bbox_input[1]+bbox_input[3]/2.0)/16.0),
                                int(bbox_input[2]/16.0),
                                int(bbox_input[3]/16.0),14,14)
#        skip5_label[:int(bbox_input[1]/16.0),:] = 0
#        skip5_label[int((bbox_input[1]+bbox_input[3])/16.0):,:] = 0
#        skip5_label[:,:int(bbox_input[0]/16.0)] = 0
#        skip5_label[:,int((bbox_input[0]+bbox_input[2])/16.0):] = 0


                
    
    # plot iou
    if dis:
        fig = plt.figure(10)        
        fig.add_subplot(2,3,1)
        plt.imshow(cv2.rectangle(img_input.copy(),
                                 (bbox_input[0],bbox_input[1]),
                                 (bbox_input[0]+bbox_input[2],
                                  bbox_input[1]+bbox_input[3]),
                                  (0,0,255), 1)[:,:,::-1])

        fig.add_subplot(2,3,2)
        plt.imshow(skip1_label) 
        fig.add_subplot(2,3,3)
        plt.imshow(skip2_label) 
        fig.add_subplot(2,3,4)
        plt.imshow(skip3_label) 
        fig.add_subplot(2,3,5)
        plt.imshow(skip4_label) 
            
            
        fig.tight_layout()

    return skip1_label, skip2_label, skip3_label, skip4_label, skip5_label
    
if __name__ == "__main__":

    print "This is a module for tracker."
    
