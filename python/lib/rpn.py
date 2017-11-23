#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:17:20 2017

@author: zhaohj
"""

import numpy as np

def bbox_target_transform(bbox, boxes):
    """
    Compute box regression targets.
    [x,y,w,h]
    """   
    
    bbox = bbox.reshape([1,4]).astype(np.float64)
    boxes = boxes.astype(np.float64)
    
    w = boxes[:, 2]
    h = boxes[:, 3]
    cx = boxes[:, 0] + 0.5 * w
    cy = boxes[:, 1] + 0.5 * h

    gt_w = bbox[:, 2]
    gt_h = bbox[:, 3]
    gt_cx = bbox[:, 0] + 0.5 * gt_w
    gt_cy = bbox[:, 1] + 0.5 * gt_h

    targets_dx = (gt_cx - cx) / w
    targets_dy = (gt_cy - cy) / h

    targets_dw = np.log(gt_w / w)
    targets_dh = np.log(gt_h / h)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

    
def bbox_pred_transform(boxes, deltas):
    """
    Add box-deltas to the default boxes.
    [x,y,w,h]
    """    
    
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2]# - boxes[:, 0] + 1.0
    heights = boxes[:, 3]# - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    """reference to faster rcnn"""
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def generate_anchors(basic, strides, cls_map_shape):
    """
    Generate anchors according to the strides of PRN layer.
    """
    
    """generate shifts and convert to col vector"""
    # compute anchor shift according to layer strides
    shift_x = np.arange(0, cls_map_shape[1]) * strides
    shift_y = np.arange(0, cls_map_shape[0]) * strides
    # generate shift grid (matrix)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 1. convert to row vector
    # 2. column-wise stack
    # 3. transpose
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    
    """generate shifted anchors and convert to column"""
    # numbers of anchor at every location
    A = basic.shape[0]
    # number of shifts (i.e. total number of anchors on the input image)
    K = shifts.shape[0]
    # add shifts to basic anchor (shape = [K,A,4])
    anchors = basic.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    # reshape to column [x1,y1,x2,y2]
    anchors = anchors.reshape((K * A, 4))
    
    return anchors

    
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    [x,y,w,h]
    """
    
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
    
    
def filter_boxes(boxes, min_size):
    """
    Remove all boxes with any side smaller than min_size.
    """
    
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep    
    
    
def nms(dets, thresh):
    """
    Pure Python NMS baseline.
    input [x,y,w,h]
    """
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]+dets[:, 0]
    y2 = dets[:, 3]+dets[:, 0]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep    
    
    

if __name__ == "__main__":

    print "This is a module for RPN."
