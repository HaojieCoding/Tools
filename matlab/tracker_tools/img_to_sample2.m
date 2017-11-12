function [samples_neg] = img_to_sample2(img,bbox,dis)

    cx = round(bbox(1)+bbox(3)/2);
    cy = round(bbox(2)+bbox(4)/2);
        
    longer = max(bbox(3),bbox(4));

    bbox_temp = [cx-longer/2, cy-longer/2, longer, longer];
    sigma = (bbox_temp(3)+bbox_temp(4))/200.0;
    samples_neg = generate_samples(img,bbox_temp,500,'whole',...
                      sigma*2,0, -1,0.001,0,0);
    rand_idx = randperm(length(samples_neg));
    samples_neg = samples_neg(rand_idx,:);
              
    samples_neg = round(samples_neg);
    samples_neg = limit_box(img, samples_neg);
     
    if dis
        imshow(img);
        for i = 1:length(samples_neg)
            rectangle('Position', samples_neg(i,:), 'EdgeColor', [1 0 0], 'Linewidth', 2);
        end
    end
        
        
    
    
    