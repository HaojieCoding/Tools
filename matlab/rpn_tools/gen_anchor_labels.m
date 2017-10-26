function [labels,dis_iou] = gen_anchor_labels(bbox_input,anchors,dis)

    iou = overlap_ratio(bbox_input,anchors);
    labels(iou==0) = -1;
    
    % set pos
    labels(iou>max(iou)*0.6) = 1;
    
    labels = reshape(labels,[size(labels,2),1]);
    
    % ignore some neg samples to keep sample balance
    temp_where = find(labels == 0);
    if length(temp_where) > 64
        ignore = randsample(temp_where,size(temp_where,1)-64);
        labels(ignore) = -1;
    end  
    
    temp_where = find(labels == 1);
    if length(temp_where) > 64
        ignore = randsample(temp_where,size(temp_where,1)-64);
        labels(ignore) = -1;
    end   
    
    
    if dis
    
        dis_iou(:,:,1) = reshape(iou(1:3:end),[16,16])';
        dis_iou(:,:,2) = reshape(iou(2:3:end),[16,16])';
        dis_iou(:,:,3) = reshape(iou(3:3:end),[16,16])';
    
%     iou = permute(iou, [2, 1, 3]);
    
        for i = 1:3
            subplot(1,3,i);
            imshow(dis_iou(:,:,i));
        end
    end

    
    
    