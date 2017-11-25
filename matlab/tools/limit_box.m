function [ bbox ] = limit_box(img, bbox)
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

for i = 1:length(bbox(:,1))
   if bbox(i,1) < 1
       bbox(i,3) = bbox(i,3) + bbox(i,1);
       bbox(i,1) = 1;
   end
   if bbox(i,2) < 1
       bbox(i,4) = bbox(i,4) + bbox(i,2);
       bbox(i,2) = 1;
   end
   
   if bbox(i,1) + bbox(i,3) > w
       bbox(i,3) = w - bbox(i,1) - 1;
   end
   if bbox(i,2) + bbox(i,4) > h
       bbox(i,4) = h - bbox(i,2) - 1;
   end

end