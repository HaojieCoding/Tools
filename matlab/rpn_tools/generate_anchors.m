function anchors = generate_anchors(basic, strides, cls_map_shape)
    
    % generate shifts
    shift_x = [0 : cls_map_shape(2)-1] * strides;
    shift_y = [0 : cls_map_shape(1)-1] * strides;
    
    shift_x = meshgrid(shift_x);
    shift_y = meshgrid(shift_y);
    
    % convert to colume
    shift_x = shift_x';
%     shift_y = shift_y';
    shift_x = shift_x(:);
    shift_y = shift_y(:);
    
    shifts = [shift_x,shift_y,shift_x,shift_y];
    
    anchors = basic;
    for i = 2:size(shifts,1)
        anchors = [anchors;basic+shifts(i,:)];
    end
    
    
    