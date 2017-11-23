function img_motion = motion_blur(img)

    % FSPECIAL('motion',LEN,THETA)
    THETA = unifrnd(0,360);
%     LEN = unifrnd(10,20);
    LEN = 20;
    PSF = fspecial('motion',LEN,THETA);
    img_motion = imfilter(img, PSF,'conv','circular');