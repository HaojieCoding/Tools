function [samples_neg] = img_to_sampleNeg(img,bbox,dis)

    side = 2*mean(bbox(3),bbox(4));
    
    num_x = round(ceil(size(img,2)/side));
    num_y = round(ceil(size(img,1)/side));
    samples_neg = zeros(num_x*num_y,4);
    samples_neg(:,3:4) = side;
    for y = 0:num_y-1
        for x = 0:num_x-1
            samples_neg(num_x*y+x+1,1) = x*side;
            samples_neg(num_x*y+x+1,2) = y*side;
        end
    end
    
    samples_neg = clip_boxes(img, samples_neg);
    samples_neg = samples_neg(samples_neg(:,3)>side/2,:);
    samples_neg = samples_neg(samples_neg(:,4)>side/2,:);
    iou = overlap_ratio(bbox,samples_neg);
    samples_neg = samples_neg(iou<=0.00,:);
    
    if dis
        imshow(img);
        for i = 1:length(samples_neg)
            rectangle('Position', samples_neg(i,:), 'EdgeColor', [1 0 0], 'Linewidth', 2);
        end
    end
        
        
    
    
    