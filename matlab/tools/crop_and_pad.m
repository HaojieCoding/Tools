function [ crop ] = crop_and_pad(img, x1, y1, x2, y2)
% 
% This function is used to crop img according to bbox.
%
%
% zhaohj, 2017
% 


mean = [ mean2(img(:,:,1)),mean2(img(:,:,2)),mean2(img(:,:,3)) ];

imgsize = size(img);
img_h = imgsize(1); 
img_w = imgsize(2);

pad_left = 0;
pad_right = 0;
pad_up = 0;
pad_down = 0;


if x1 < 1
    pad_left = round(1-x1);
    x1 = 1;
end
if y1 < 1
    pad_up = round(1-y1);
    y1 = 1;
end
if x2 > img_w
    pad_right = round(x2 - img_w + 1);
    x2 = img_w;
end
if y2 > img_h
    pad_down = round(y2 - img_h + 1);
    y2 = img_h;
end

crop = img(round(y1):round(y2),round(x1):round(x2),:);

% pad
if pad_left ~= 0
    pad = ones([size(crop,1),pad_left,3]);
    pad(:,:,1) = mean(1);
    pad(:,:,2) = mean(2);
    pad(:,:,3) = mean(3);
    crop = [pad,crop];
% fprintf('%d %d %d \n',size(crop,1),size(crop,2),size(crop,3));
% fprintf('%d %d %d \n',size(pad,1),size(pad,2),size(pad,3));
end
if pad_right ~= 0
    pad = ones([size(crop,1),pad_right,3]);
    pad(:,:,1) = mean(1);
    pad(:,:,2) = mean(2);
    pad(:,:,3) = mean(3);
    crop = [crop,pad];
% fprintf('%d %d %d \n',size(crop,1),size(crop,2),size(crop,3));
% fprintf('%d %d %d \n',size(pad,1),size(pad,2),size(pad,3));
end
if pad_up ~= 0
    pad = ones([pad_up,size(crop,2),3]);
    pad(:,:,1) = mean(1);
    pad(:,:,2) = mean(2);
    pad(:,:,3) = mean(3);
    crop = [pad;crop];
% fprintf('%d %d %d \n',size(crop,1),size(crop,2),size(crop,3));
% fprintf('%d %d %d \n',size(pad,1),size(pad,2),size(pad,3));
end
if pad_down ~= 0
    pad = ones([pad_down,size(crop,2),3]);
    pad(:,:,1) = mean(1);
    pad(:,:,2) = mean(2);
    pad(:,:,3) = mean(3);
    crop = [crop;pad];
% fprintf('%d %d %d \n',size(crop,1),size(crop,2),size(crop,3));
% fprintf('%d %d %d \n',size(pad,1),size(pad,2),size(pad,3));
end
