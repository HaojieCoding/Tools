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
side_1 = mean(bboxes(i,3),bboxes(i,4));
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
   if bboxes(i,3) < 0 || bboxes(i,4) < 0
       bboxes(i,1:2) = [w/2-side_1/2,h/2-side_1/2];
       bboxes(i,3:4) = side_1;
   end
end