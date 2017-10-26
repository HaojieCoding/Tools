function targets = bbox_target_transdorm(bbox,boxes)

    w = boxes(:,3);
    h = boxes(:,4);
    cx = boxes(:,1) + w/2;
    cy = boxes(:,2) + h/2;
    
    gt_w = bbox(:,3);
    gt_h = bbox(:,4);
    gt_cx = bbox(:,1) + gt_w/2;
    gt_cy = bbox(:,2) + gt_h/2;
    
    targets_dx = (gt_cx - cx)./w;
    targets_dy = (gt_cy - cy)./h;
    
    targets_dw = log(gt_w./w);
    targets_dh = log(gt_h./h);
    
    targets = double([targets_dx,targets_dy,targets_dw,targets_dh]);
%     targets = reshape(targets,[4,size(targets,1)/4]);
%     targets = targets';
    