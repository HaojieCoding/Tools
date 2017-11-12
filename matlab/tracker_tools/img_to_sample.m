function [img_sample,bbox_sample,origin] = img_to_sample(img,bbox,flip,dis)

    cx = round(bbox(1)+bbox(3)/2);
    cy = round(bbox(2)+bbox(4)/2);
    
    f = unifrnd(1.5,2.8,1);
    
    longer = max(bbox(3),bbox(4));
    side_w = round(longer*f);
    side_h = round(longer*f);
    
    % rand xy
    rand_x = unifrnd(-round(bbox(3)/2*(f-1)), round(bbox(3)/2*(f-1)), 1);
    rand_y = unifrnd(-round(bbox(4)/2*(f-1)), round(bbox(4)/2*(f-1)), 1);
    
    temp_x1 = round(cx + rand_x - side_w/2);
    temp_y1 = round(cy + rand_y - side_h/2);
    temp_x2 = round(cx + rand_x + side_w/2)+1;
    temp_y2 = round(cy + rand_y + side_h/2)+1;
    origin = [temp_x1,temp_y1];
    
    temp_box = [temp_x1,temp_y1,temp_x2-temp_x1,temp_y2-temp_y1];
    
    sample = temp_box;
    
    
    img_sample = crop_and_pad(img,sample(1),sample(2),sample(1)+sample(3),sample(2)+sample(4));                       
                          

	bbox_sample = [1,1,1,1];
    bbox_sample(1) = round(bbox(1) - sample(1));
    bbox_sample(2) = round(bbox(2) - sample(2));
    bbox_sample(3) = round(bbox(3));
    bbox_sample(4) = round(bbox(4));
    
    bbox_sample = limit_box(img_sample,bbox_sample);
    
    % random bright
    img_temp = img_sample;
    img_temp = round(img_temp + max(-10,unifrnd(-50,20)) ); 
    for c = 1:3
        for x = 1:size(img_temp,1)
            for y = 1:size(img_temp,2)
                if img_temp(x,y,c) > 255
                    img_temp(x,y,c) = 255;
                end
                if img_temp(x,y,c) < 0
                    img_temp(x,y,c) = 0;
                end
            end
        end
    end
    img_sample = img_temp;
    
    %random flip
    if flip
        if floor(unifrnd(0,2))
            img_sample = fliplr(img_sample);
            bbox_sample(2) = bbox_sample(2);
            bbox_sample(1) = size(img_sample,2) - bbox_sample(1) - bbox_sample(3);
        end
    end
    
    % random blur
    if floor(unifrnd(0,2))
        img_sample = imfilter(img_sample,fspecial('gaussian',[3,3],3));
    end
     
    if dis
        imshow(img_sample);
%         rectangle('Position', bbox_sample, 'EdgeColor', [1 0 0], 'Linewidth', 2);
    end
        
        
    
    
    