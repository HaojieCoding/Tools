function [samples_buf,bbox_buf] = img_to_samplePos(img,bbox,f_list,dis)

    cx = round(bbox(1)+bbox(3)/2);
    cy = round(bbox(2)+bbox(4)/2);
        
    longer = max(bbox(3),bbox(4));
    
    img_buf = {};
    samples_buf = [];
    bbox_buf = [];
    
    for f = f_list
        side = round(longer*f);
        
        off_x = round( (side-bbox(3))/2.0 );
        off_y = round( (side-bbox(4))/2.0 );
        
        off = [     0,     0,     0,     0;...
                off_x,     0, off_x,     0;...
               -off_x,     0,-off_x,     0;...
                    0, off_y,     0, off_y;...
                    0,-off_y,     0,-off_y;...
                off_x/2,     0, off_x/2,     0;...
               -off_x/2,     0,-off_x/2,     0;...
                    0, off_y/2,     0, off_y/2;...
                    0,-off_y/2,     0,-off_y/2;...
                off_x/4,     0, off_x/4,     0;...
               -off_x/4,     0,-off_x/4,     0;...
                    0, off_y/4,     0, off_y/4;...
                    0,-off_y/4,     0,-off_y/4;...
                                                ];
                
        % x1, y1, x2, y2
        crop = [cx-side/2, cy-side/2, cx+side/2, cy+side/2];
        crop = repmat(crop,13,1);
        crop = crop + off;
        
        bbox_sample = repmat(bbox,13,1);
                
        
        % to [x,y,w,h]
        crop(:,3:4) = crop(:,3:4) - crop(:,1:2);
        
        for i = 1:size(crop,1)
            img_sample = crop_and_pad(img,...
                                      round(crop(i,1)), round(crop(i,2)),...
                                      round(crop(i,1)+crop(i,3)),...
                                      round(crop(i,2)+crop(i,4)));
            % bbox_sample(bbox) to [x,y,x,y]
            bbox_sample(i,3) = bbox_sample(i,3) + bbox_sample(i,1);
            bbox_sample(i,4) = bbox_sample(i,4) + bbox_sample(i,2);
            % subtract l-t point of crop box
            bbox_sample(i,1) = max(1, min(bbox_sample(i,1)-crop(i,1), side-1));
            bbox_sample(i,2) = max(1, min(bbox_sample(i,2)-crop(i,2), side-1));
            bbox_sample(i,3) = max(1, min(bbox_sample(i,3)-crop(i,1), side-1));
            bbox_sample(i,4) = max(1, min(bbox_sample(i,4)-crop(i,2), side-1));
            % to [x,y,w,h]
            bbox_sample(i,3) = bbox_sample(i,3)-bbox_sample(i,1)-1;
            bbox_sample(i,4) = bbox_sample(i,4)-bbox_sample(i,2)-1;
            
            fw = size(img_sample,2)/224.0;
            fh = size(img_sample,1)/224.0;
            
            bbox_sample(i,1:2:4) = bbox_sample(i,1:2:4)./fw;
            bbox_sample(i,2:2:4) = bbox_sample(i,2:2:4)./fh;
            
            img_sample = imresize(img_sample,[224,224]);
            
            
            img_buf = [img_buf, img_sample];
            samples_buf = [samples_buf;crop(i,:)];
            
        end
        bbox_buf = [bbox_buf; bbox_sample];
    end
    
    bbox_buf = round(bbox_buf);
        
%     bbox_buf = clip_boxes(img, bbox_buf);
    
     
    if dis
        imshow(img);
        for i = 1:length(samples_buf)
            rectangle('Position', samples_buf(i,:), 'EdgeColor', [1 0 0], 'Linewidth', 2);
        end
    end
        
        
    
    
    