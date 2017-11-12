function mask = gen_mask(feat)

%     mask = ones(241,241);
%     mask = mask*0.3;
%     
%     mask(bbox(2)-29:bbox(2)+bbox(4)+29,...
%         bbox(1)-29:bbox(1)+bbox(3)+29) = 0.8;
%     
% 	mask(bbox(2):bbox(2)+bbox(4),...
%         bbox(1):bbox(1)+bbox(3)) = 1.0;

    bw = im2bw(feat/255,0.7);
    SE=strel('square',9);
    mask = imerode(bw,SE);
    mask = imerode(mask,SE);
    mask = imerode(mask,SE);
    mask = imerode(mask,SE);

    
    SE=strel('square',9);
    mask = imdilate(mask,SE);
    mask = imdilate(mask,SE);
    mask = imdilate(mask,SE);
    mask = imdilate(mask,SE);
    mask = imdilate(mask,SE);
    mask = imdilate(mask,SE);
    mask = imdilate(mask,SE);

   