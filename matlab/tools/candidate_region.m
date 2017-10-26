function [ candidate ] = candidate_region(bbox, drift, img)
% 
% This function is used to generate train samples.
%
% sigma1: std for center of gt_box
% sigma2: std for h and w of gt_box
%
% zhaohj, 2017
% 

imgsize = size(img);
h = imgsize(1); 
w = imgsize(2);

candidate = zeros(9,4);

region_w = bbox(3);
region_h = bbox(4);

x_l = bbox(1)-bbox(3)*drift;
x_c = bbox(1);
x_r = bbox(1)+bbox(3)*drift;

y_u = bbox(2)-bbox(4)*drift;
y_c = bbox(2);
y_d = bbox(2)+bbox(4)*drift;

if x_l < 1
    x_l = 1;
end

if y_u < 1
    y_u = 1;
end
    
candidate(1,:) = [x_l, y_u, region_w, region_h];
candidate(2,:) = [x_c, y_u, region_w, region_h];
candidate(3,:) = [x_r, y_u, region_w, region_h];

candidate(4,:) = [x_l, y_c, region_w, region_h];
candidate(5,:) = [x_c, y_c, region_w, region_h];
candidate(6,:) = [x_r, y_c, region_w, region_h];

candidate(7,:) = [x_l, y_d, region_w, region_h];
candidate(8,:) = [x_c, y_d, region_w, region_h];
candidate(9,:) = [x_r, y_d, region_w, region_h];

if (x_r + region_w) > (w-1)
    candidate(3,3) = w - x_r;
    candidate(6,3) = w - x_r;
    candidate(9,3) = w - x_r;
end

if (y_d + region_h) > (h-1)
    candidate(3,4) = h - y_d;
    candidate(6,4) = h - y_d;
    candidate(9,4) = h - y_d;
end
        