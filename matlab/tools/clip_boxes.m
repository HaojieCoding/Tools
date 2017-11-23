function [ bboxes ] = clip_boxes(img, bboxes)
% 
% This function is used to limit bbox size.
%
% sigma1: std for center of gt_box
% sigma2: std for h and w of gt_box
%
% zhaohj, 2017
% 

imgsize = size(img);
h = imgsize(1); 
w = imgsize(2);

for i = 1:length(bboxes(:,1))
   if bboxes(i,1) < 1
       bboxes(i,3) = bboxes(i,3) + bboxes(i,1);
       bboxes(i,1) = 1;
   end
   if bboxes(i,2) < 1
       bboxes(i,4) = bboxes(i,4) + bboxes(i,2);
       bboxes(i,2) = 1;
   end
   
   if bboxes(i,1) + bboxes(i,3) > w
       bboxes(i,3) = w - bboxes(i,1) - 1;
   end
   if bboxes(i,2) + bboxes(i,4) > h
       bboxes(i,4) = h - bboxes(i,2) - 1;
   end

end