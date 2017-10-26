function bbox = find_max_contour(input_map)

    bw = im2bw(input_map/255,0.7);

    labeled = bwlabel(bw);
    
    if max(max(labeled)) > 0
        measurements = regionprops(labeled,'Area');
        allarea = [measurements.Area];

        idx_max = find(allarea == max(allarea));
        [y,x] = find(labeled == idx_max);

        x1 = min(x);
        x2 = max(x);
        y1 = min(y);
        y2 = max(y);

        bbox=[x1,y1,x2-x1,y2-y1];
    else
        bbox=[];
    end
    

%    rectangle('Position', bbox, 'EdgeColor', [0 0 1], 'Linewidth', 3); 
    