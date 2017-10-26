function side = side_test(side)
    if side < 60
        side = 70;
    elseif side>=60 && side <80
        side = 100;
    elseif side>=80 && side<110
        side = 140;
    elseif side>=110 && side<140
        side = 180;
    elseif side>=140 && side<180
        side = 220;
    elseif side>=180
        side = 270;
    end
    
    side = round(side/2);