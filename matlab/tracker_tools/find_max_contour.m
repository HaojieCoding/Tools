function bbox = find_max_contour(input_map)

    labeled = bwlabel(input_map);
    
    if max(max(labeled)) > 0
        measurements = regionprops(labeled,'Area');
        allarea = [measurements.Area];

        idx_max = find(allarea == max(allarea));
        
        if length(idx_max) == 1
            [y,x] = find(labeled == idx_max);
        elseif length(idx_max) > 1
            [y,x] = find(labeled == idx_max(1));
        end

        x1 = min(x);
        x2 = max(x);
        y1 = min(y);
        y2 = max(y);

        bbox=[x1,y1,x2-x1,y2-y1];
    else
        bbox=[];
    end
    

%    rectangle('Position', bbox, 'EdgeColor', [0 0 1], 'Linewidth', 3); 
    