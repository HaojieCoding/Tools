function [ samples ] = generate_samples(img, gt_box, num, type, sigma1, sigma2,clip)
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

% center_x center_y
center_x = gt_box(1)+gt_box(3)/2;
center_y = gt_box(2)+gt_box(4)/2;

% n copy  for store samples
sample = [0, 0, 0, 0];
samples = repmat(sample, [num, 1]); 


switch (type)
    case 'around'
        samples(:,1) = round( normrnd(center_x, sigma1, [num,1]));
        samples(:,2) = round( normrnd(center_y, sigma1, [num,1]));
        
        samples(:,3) = round( normrnd(gt_box(3), sigma2, [num,1]) );
        samples(:,4) = round( normrnd(gt_box(4), sigma2, [num,1]) );    
        
        samples(:,1) = samples(:,1) - round(samples(:,3)/2);
        samples(:,2) = samples(:,2) - round(samples(:,4)/2);
    case 'whole'
        samples(:,1) = round( unifrnd(1,w-gt_box(3),num,1) );
        samples(:,2) = round( unifrnd(1,h-gt_box(4),num,1) );
        
        samples(:,3) = round( normrnd(gt_box(3), sigma2, [num,1]) );
        samples(:,4) = round( normrnd(gt_box(4), sigma2, [num,1]) );    
end
 
samples(:,3) = round(max( gt_box(3)*0.4, min( gt_box(3)*1.6, samples(:,3) ) ));
samples(:,4) = round(max( gt_box(4)*0.4, min( gt_box(4)*1.6, samples(:,4) ) ));

if clip
    samples = limit_box(img, samples);
end
