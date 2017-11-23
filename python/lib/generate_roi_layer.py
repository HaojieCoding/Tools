import caffe
import numpy as np
import rpn
import utils
import math

class GenerateRoiLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        pass
    
    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        """
        bottom[0]: pred_cls (shape: [18,16,16])
        bottom[1]: bbox (shape: [4,])
        
        top[0]: rois (shape: [10,5])
        top[1]: cls_labels (shape: [10,1])
        """
        bbox = bottom[1]
        pred_cls = bottom[0][9:,:,:].transpose(1,2,0).reshape(-1,1).ravel()

        anchors_crop = _generate_anchors(bbox, 16,(16,16))        
        sortlist = np.argsort(pred_cls)[::-1]

        rois = anchors_crop[sortlist[:10]]

        rois = np.hstack((np.zeros([10,1]),rois))
        
        top[0] = rois

            
        cls_labels = np.zeros([10,1])
        iou = utils.overlap_ratio(bbox,rois[:,1:])
        cls_labels[iou > 0.6] = 1
        
        rpn.nms(np.hstack((anchors_crop[sortlist[:30]],pred_cls[sortlist[:30]])),0.5)
        
        
def _generate_anchors(gtBox, strides, cls_map_shape):
    basic_anchor = np.zeros([9,4])

    temp = min(gtBox[2],gtBox[3])
    temp = math.floor(temp/2.0)
    temp_s = math.floor(temp*0.5)
    temp_m = math.floor(temp*1.5)
    
    basic_anchor[3:6,:] = np.array([[-temp_s-1,-temp_s-1,temp_s,temp_s],
                                    [-temp-1  ,-temp-1  ,temp,temp],
                                    [-temp_m-1,-temp_m-1,temp_m,temp_m]])
    
    temp = math.floor(temp*math.sqrt(0.5))
    temp_s = math.floor(temp*0.5)
    temp_m = math.floor(temp*1.5)
    
    basic_anchor[0:3,:] = np.array([[-2*temp_s-1,-temp_s-1,2*temp_s,temp_s],
                                    [-2*temp-1  ,-temp-1  ,2*temp  ,temp  ],
                                    [-2*temp_m-1,-temp_m-1,2*temp_m,temp_m]])
    
    basic_anchor[6:9,:] = np.array([[-temp_s-1,-2*temp_s-1,temp_s,2*temp_s],
                                    [-temp-1  ,-2*temp-1  ,temp  ,2*temp  ],
                                    [-temp_m-1,-2*temp_m-1,temp_m,2*temp_m]])
    
    
    anchors = rpn.generate_anchors(basic_anchor, strides, cls_map_shape)
    
    # clip anchors
    anchors_crop = anchors
    # [x,y,w,h]
    anchors_crop[:,2] -= anchors_crop[:,0]
    anchors_crop[:,3] -= anchors_crop[:,1]
    anchors_crop = rpn.clip_boxes(anchors_crop,(241,241,3))
    
    return anchors_crop