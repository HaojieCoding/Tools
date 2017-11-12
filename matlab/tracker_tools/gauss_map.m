function [gauss_map] = gauss_map(cx,cy,w,h,img_w,img_h)

    mu_w = cx;
    mu_h = cy;
    
    sigma_w = w/4.0;
    sigma_h = h/4.0;
    
    x_w = linspace(1,img_w,img_w);
    x_h = linspace(1,img_h,img_h);  
    
    y_w = normpdf(x_w, mu_w, sigma_w);
    y_h = normpdf(x_h, mu_h, sigma_h);
    
    gauss_map = y_h' * y_w;
    gauss_map = gauss_map/max(gauss_map(:));