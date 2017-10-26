function iou = overlap_ratio(gt_box, box)
% 
% This is a function of MDNet, which is used to compute overlap ratio.
%
% OVERLAP_RATIO
% Compute the overlap ratio between two rectangles
%
% Hyeonseob Nam, 2015
% 

inter_area = rectint(box,gt_box); % matlab compute rectangle intersection area
union_area = box(:,3).*box(:,4) + gt_box(:,3).*gt_box(:,4) - inter_area;

iou = inter_area./union_area;
end