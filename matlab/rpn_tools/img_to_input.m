function [img_input,bbox_input,factor_w,factor_h] = ...
                img_to_input(img_sample,bbox_sample,dis)
    
            
    factor_w = 241.0/size(img_sample,2);
    factor_h = 241.0/size(img_sample,1);
    img_input = imresize(img_sample,[241,241]);
    
    bbox_input(1) = ceil(bbox_sample(1)*factor_w) +1;
    bbox_input(3) = ceil(bbox_sample(3)*factor_w) +1;
    bbox_input(2) = ceil(bbox_sample(2)*factor_h);
    bbox_input(4) = ceil(bbox_sample(4)*factor_h);
    bbox_input = limit_box(img_input,bbox_input);
    if dis
        imshow(img_input);
        rectangle('Position', bbox_input, 'EdgeColor', [1 0 0], 'Linewidth', 2);
    end
