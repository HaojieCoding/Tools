function [samples_neg] = img_to_sampleNeg(img,bbox,dis)

    side = 2*max(bbox(3),bbox(4));
    
    num_x = round(ceil(size(img,2)/side));
    num_y = round(ceil(size(img,1)/side));
    samples_temp = zeros(num_x*num_y,4);
    samples_temp(:,3:4) = side;
    for y = 0:num_y-1
        for x = 0:num_x-1
            samples_temp(num_x*y+x+1,1) = x*side;
            samples_temp(num_x*y+x+1,2) = y*side;
        end
    end
    
    samples_temp = clip_boxes(img, samples_temp);
    samples_temp = samples_temp(samples_temp(:,3)>side/2,:);
    samples_temp = samples_temp(samples_temp(:,4)>side/2,:);
    iou = overlap_ratio(bbox,samples_temp);
    samples_temp = samples_temp(iou<=0.00,:);
    
    samples_neg = samples_temp;
    
    % crop small
    side = 1*max(bbox(3),bbox(4));
    
    num_x = round(ceil(size(img,2)/side));
    num_y = round(ceil(size(img,1)/side));
    samples_temp = zeros(num_x*num_y,4);
    samples_temp(:,3:4) = side;
    for y = 0:num_y-1
        for x = 0:num_x-1
            samples_temp(num_x*y+x+1,1) = x*side;
            samples_temp(num_x*y+x+1,2) = y*side;
        end
    end
    
    samples_temp = clip_boxes(img, samples_temp);
    samples_temp = samples_temp(samples_temp(:,3)>side/2,:);
    samples_temp = samples_temp(samples_temp(:,4)>side/2,:);
    iou = overlap_ratio(bbox,samples_temp);
    samples_temp = samples_temp(iou<=0.00,:);
    
    
    samples_neg = [samples_neg; samples_temp];
    
    if dis
        imshow(img);
        for i = 1:size(samples_neg,1)
            rectangle('Position', samples_neg(i,:), 'EdgeColor', [1 0 0], 'Linewidth', 2);
        end
    end
        
        
    
    
    