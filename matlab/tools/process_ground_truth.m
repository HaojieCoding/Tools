function gt_box = process_ground_truth(region)
if length(region) == 8
    x = round(min(region(1:2:end)));
    y = round(min(region(2:2:end)));
    w = round(max(region(1:2:end)) - min(region(1:2:end)));
    h = round(max(region(2:2:end)) - min(region(2:2:end)));
else
    x = round(region(1));
    y = round(region(2));
    w = round(region(3));
    h = round(region(4));
end;
gt_box = [x y w h];

end
