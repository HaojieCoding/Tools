function anchors = set_anchors()
    basic_anchor = ...
        [-30 ,-30 ,29 ,29;...
        -50 ,-50 ,49 ,49;...
        -70 ,-70 ,69 ,69 ];
    
    anchors = generate_anchors(basic_anchor,16, [16,16]);
    
    for i = 1:size(anchors,1)
        if anchors(i,1) < 0 || anchors(i,2) < 0 ||...
                anchors(i,3) > 241 || anchors(i,4) > 241
            anchors(i,:) = [0,0,0,0];
        end
    end
    
    anchors(:,3) = anchors(:,3) - anchors(:,1) + 1;
    anchors(:,4) = anchors(:,4) - anchors(:,2) + 1;
