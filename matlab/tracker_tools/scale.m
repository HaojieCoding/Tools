function [x,y,w,h] = scale(pred, t)

    [y,x] = find(pred > 0.5);
    cx = mean(x);
    cy = mean(y);

    for f = 1:50
        f2 = min(cy+(f+1)*2, 223);
        f1 = min(cy+f*2, 223);
        up = (sum(sum(pred(cy:f2, cx-2:cx+2))) - ...
              sum(sum(pred(cy:f1, cx-2:cx+2)))) / ...
              sum(sum(pred(cy:f1, cx-2:cx+2)));
        if up < t
            break;
        end
    end
    h1 = f*2;
    h1 = min(cy+h1, 223)-cy;

    for f = 1:50
        f2 = max(cy-(f+1)*2, 0);
        f1 = max(cy-f*2, 0);
        up = (sum(sum(pred(f2:cy, cx-2:cx+2))) - ...
              sum(sum(pred(f1:cy, cx-2:cx+2)))) / ...
              sum(sum(pred(f1:cy, cx-2:cx+2)));
        if up < t
            break;
        end
    end
    h2 = f*2;
    h2 = cy-max(cy-h2, 0);

    h = h1 + h2;


    for f = 1:50
        f2 = min(cx+(f+1)*2, 223);
        f1 = min(cx+f*2, 223);
        up = (sum(sum(pred(cy-2:cy+2, cx:f2))) - ...
              sum(sum(pred(cy-2:cy+2, cx:f1)))) / ...
              sum(sum(pred(cy-2:cy+2, cx:f1)));
        if up < t
            break;
        end
    end
    w1 = f*2;
    w1 = min(cx+w1, 223)-cx;

    for f = 1:50
        f2 = max(cx-(f+1)*2, 0);
        f1 = max(cx-f*2, 0);
        up = (sum(sum(pred(cy-2:cy+2, f2:cx))) - ...
              sum(sum(pred(cy-2:cy+2, f1:cx)))) / ...
              sum(sum(pred(cy-2:cy+2, f1:cx)));
        if up < t
            break;
        end
    end
    w2 = f*2;
    w2 = cx-max(cx-w2, 0);

    w = w1 + w2;
 
    x = cx-w2;
    y = cy-h2;
    
    
end