function [img_sample,bbox_sample,origin] = img_to_sample(img,bbox,flip,dis)

    cx = round(bbox(1)+bbox(3)/2);
    cy = round(bbox(2)+bbox(4)/2);
    
    rand = unifrnd(1.02,1.5,1);
    
    side_w = round(bbox(3)*rand);
    side_h = round(bbox(4)*rand);
    
    temp_x1 = round(cx - side_w);
    temp_y1 = round(cy - side_h);
    temp_x2 = round(cx + side_w)+1;
    temp_y2 = round(cy + side_h)+1;
    origin = [temp_x1,temp_y1];
    
    temp_box = [temp_x1,temp_y1,temp_x2-temp_x1,temp_y2-temp_y1];
    
    sample = generate_samples(img,temp_box,1,'around',...
                                round((bbox(2)+bbox(3))/2*0.17),0,false);
    
    
    img_sample = crop_and_pad(img,sample(1),sample(2),sample(1)+sample(3),sample(2)+sample(4));                       
                          

	bbox_sample = [1,1,1,1];
    bbox_sample(1) = round(bbox(1) - sample(1));
    bbox_sample(2) = round(bbox(2) - sample(2));
    bbox_sample(3) = round(bbox(3));
    bbox_sample(4) = round(bbox(4));
    
    bbox_sample = limit_box(img_sample,bbox_sample);
    
    % random bright
    img_temp = img_sample;
    img_temp = round(img_temp + max(-20,unifrnd(-100*0.6,100*0.7)) ); 
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
        rectangle('Position', bbox_sample, 'EdgeColor', [1 0 0], 'Linewidth', 2);
    end
        
        
    
    
    