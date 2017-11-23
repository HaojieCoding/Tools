function [skip1,skip2,skip3,skip4,skip5]...
= gen_skip_labels(img_input,bbox,gauss,dis)
        

    bbox = round(bbox)+1;

    skip1 = zeros(224,224);
    skip1(bbox(2):bbox(2)+bbox(4),bbox(1):bbox(1)+bbox(3)) = 1;
  
    
    skip2 = zeros(112,112);
    skip2(ceil(bbox(2)/2):ceil(bbox(2)/2+bbox(4)/2),...
          ceil(bbox(1)/2):ceil(bbox(1)/2+bbox(3)/2)) = 1;


    skip3 = zeros(56,56);
    skip3(ceil(bbox(2)/4):ceil(bbox(2)/4+bbox(4)/4),...
          ceil(bbox(1)/4):ceil(bbox(1)/4+bbox(3)/4)) = 1;

    skip4 = zeros(28,28);
    skip4(ceil(bbox(2)/8):ceil(bbox(2)/8+bbox(4)/8),...
          ceil(bbox(1)/8):ceil(bbox(1)/8+bbox(3)/8)) = 1;

    skip5 = zeros(14,14);
    skip5(ceil(bbox(2)/16):ceil(bbox(2)/16+bbox(4)/16),...
          ceil(bbox(1)/16):ceil(bbox(1)/16+bbox(3)/16)) = 1;

    if gauss
        skip1 = gauss_map(ceil(bbox(1)+bbox(3)/2),...
                          ceil(bbox(2)+bbox(4)/2),...
                          bbox(3),bbox(4),224,224);
                      
        skip2 = gauss_map(ceil((bbox(1)+bbox(3)/2)/2),...
                          ceil((bbox(2)+bbox(4)/2)/2),...
                          ceil(bbox(3)/2),...
                          ceil(bbox(4)/2),112,112);
                      
        skip3 = gauss_map(ceil((bbox(1)+bbox(3)/2)/4),...
                          ceil((bbox(2)+bbox(4)/2)/4),...
                          ceil(bbox(3)/4),...
                          ceil(bbox(4)/4),56,56);
                      
        skip4 = gauss_map(ceil((bbox(1)+bbox(3)/2)/8),...
                          ceil((bbox(2)+bbox(4)/2)/8),...
                          ceil(bbox(3)/8),...
                          ceil(bbox(4)/8),28,28);
                      
        skip5 = gauss_map(ceil((bbox(1)+bbox(3)/2)/16),...
                          ceil((bbox(2)+bbox(4)/2)/16),...
                          ceil(bbox(3)/16),...
                          ceil(bbox(4)/16),14,14);
                      
    end
    
    if bbox(4) == 0
        skip1 = zeros(224,224);
        skip2 = zeros(112,112);
        skip3 = zeros(56,56);
        skip4 = zeros(28,28);
        skip5 = zeros(14,14);
    end

        
    if dis
        figure(1);
        subplot(2,3,1);
        imshow(skip5);
        subplot(2,3,2);
        imshow(skip4);
        subplot(2,3,3);
        imshow(skip3);
        subplot(2,3,4);
        imshow(skip2);
        subplot(2,3,5);
        imshow(skip1);
    end
                      


    