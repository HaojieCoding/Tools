function [img_test,bbox_test,origin] = img_to_test(img,bbox,dis)

    cx = round(bbox(1)+bbox(3)/2);
    cy = round(bbox(2)+bbox(4)/2);
    
    bbox(3) = round(bbox(3)*1.04);
    bbox(4) = round(bbox(4)*1.04);
    
    side_w = side_test(bbox(3));
    side_h = side_test(bbox(4));
    
    temp_x1 = round(cx - side_w);
    temp_y1 = round(cy - side_h);
    temp_x2 = round(cx + side_w)+1;
    temp_y2 = round(cy + side_h)+1;
    origin = [temp_x1,temp_y1];
   
    
    
    img_test = crop_and_pad(img,temp_x1,temp_y1,temp_x2,temp_y2);
    
	bbox_test = [1,1,1,1];
    bbox_test(1) = round(bbox(1) - temp_x1);
    bbox_test(2) = round(bbox(2) - temp_y1);
    bbox_test(3) = round(bbox(3));
    bbox_test(4) = round(bbox(4));
     
    if dis
        imshow(img_test);
        rectangle('Position', bbox_test, 'EdgeColor', [1 0 0], 'Linewidth', 2);
    end
        
    
    
    