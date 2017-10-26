#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:25:47 2017

@author: zhaohj
"""

import xml.etree.cElementTree as ET
import os

#%% get img list
f = []
for (dirpath, dirnames, img_list) in os.walk('/home/zhaohj/Documents/VOC/VOC2012/Annotations'):
    f.extend(img_list)
    img_list.sort(key = str.lower)
    break        

#%% parse xml
for i in range(len(img_list)):
    tree = ET.ElementTree(file='/home/zhaohj/Documents/VOC/VOC2012/Annotations/'+img_list[i][:-4]+'.xml')
    
    root = tree.getroot()
    root.tag, root.attrib
    
    name = []
    for elem in tree.iterfind('object/name'):
        name.append(elem.text)
        
    xmax = []
    for elem in tree.iterfind('object/bndbox/xmax'):
        xmax.append(elem.text)
        
    ymax = []
    for elem in tree.iterfind('object/bndbox/ymax'):
        ymax.append(elem.text)
        
    xmin = []
    for elem in tree.iterfind('object/bndbox/xmin'):
        xmin.append(elem.text)
    
    ymin = []
    for elem in tree.iterfind('object/bndbox/ymin'):
        ymin.append(elem.text)    

    """write gt text"""
    with open('/home/zhaohj/Documents/VOC/VOC2012/GroundTruth/'+img_list[i][:-4]+'.txt','w+') as f:
        for i in range(len(name)):
            string = name[i]+' '+xmin[i]+' '+ymin[i]+' '+str(float(xmax[i])-float(xmin[i]))+' '+str(float(ymax[i])-float(ymin[i]))
            f.write(string+'\n')
        
        

        
        