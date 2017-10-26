function [in,out,target] = rpn_weights_and_target(labels,targets)
    weight_in_rpn = zeros(16,16,12);
    weight_out_rpn = zeros(16,16,12);
    target = zeros(16,16,12);

    if ~isempty(find(labels(:,:,1)>-1))
        weight_in_rpn(find(labels(:,:,1)>-1)) = 1;
        weight_in_rpn(find(labels(:,:,1)>-1)+16*16*1) = 1;
        weight_in_rpn(find(labels(:,:,1)>-1)+16*16*2) = 1;
        weight_in_rpn(find(labels(:,:,1)>-1)+16*16*3) = 1;
    end
    if ~isempty(find(labels(:,:,2)>-1))
        weight_in_rpn(find(labels(:,:,2)>-1)+16*16*4) = 1;
        weight_in_rpn(find(labels(:,:,2)>-1)+16*16*5) = 1;
        weight_in_rpn(find(labels(:,:,2)>-1)+16*16*6) = 1;
        weight_in_rpn(find(labels(:,:,2)>-1)+16*16*7) = 1;
    end
    if ~isempty(find(labels(:,:,3)>-1))
        weight_in_rpn(find(labels(:,:,3)>-1)+16*16*8) = 1;
        weight_in_rpn(find(labels(:,:,3)>-1)+16*16*9) = 1;
        weight_in_rpn(find(labels(:,:,3)>-1)+16*16*10) = 1;
        weight_in_rpn(find(labels(:,:,3)>-1)+16*16*11) = 1;
    end

    if ~isempty(find(labels(:,:,1)==1))
        weight_out_rpn(find(labels(:,:,1)==1)) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,1)==1)+16*16*1) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,1)==1)+16*16*2) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,1)==1)+16*16*3) = 1/length(labels(labels==1));
    end
    if ~isempty(find(labels(:,:,2)==1))
        weight_out_rpn(find(labels(:,:,2)==1)+16*16*4) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,2)==1)+16*16*5) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,2)==1)+16*16*6) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,2)==1)+16*16*7) = 1/length(labels(labels==1));
    end
    if ~isempty(find(labels(:,:,3)==1))
        weight_out_rpn(find(labels(:,:,3)==1)+16*16*8) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,3)==1)+16*16*9) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,3)==1)+16*16*10) = 1/length(labels(labels==1));
        weight_out_rpn(find(labels(:,:,3)==1)+16*16*11) = 1/length(labels(labels==1));
    end
    
    if ~isempty(find(labels(:,:,1)==0))
        weight_out_rpn(find(labels(:,:,1)==0)) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,1)==0)+16*16*1) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,1)==0)+16*16*2) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,1)==0)+16*16*3) = 1/length(labels(labels==0));
    end
    if ~isempty(find(labels(:,:,2)==0))
        weight_out_rpn(find(labels(:,:,2)==0)+16*16*4) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,2)==0)+16*16*5) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,2)==0)+16*16*6) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,2)==0)+16*16*7) = 1/length(labels(labels==0));
    end
    if ~isempty(find(labels(:,:,3)==0))
        weight_out_rpn(find(labels(:,:,3)==0)+16*16*8) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,3)==0)+16*16*9) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,3)==0)+16*16*10) = 1/length(labels(labels==0));
        weight_out_rpn(find(labels(:,:,3)==0)+16*16*11) = 1/length(labels(labels==0));
    end
    
    in = weight_in_rpn;
    out = weight_out_rpn;
    
    
    
    
    target(:,:,1) = reshape(targets(1:3:end,1),[16,16])';
    target(:,:,2) = reshape(targets(1:3:end,2),[16,16])';
    target(:,:,3) = reshape(targets(1:3:end,3),[16,16])';
    target(:,:,4) = reshape(targets(1:3:end,4),[16,16])';
    
    target(:,:,5) = reshape(targets(2:3:end,1),[16,16])';
    target(:,:,6) = reshape(targets(2:3:end,2),[16,16])';
    target(:,:,7) = reshape(targets(2:3:end,3),[16,16])';
    target(:,:,8) = reshape(targets(2:3:end,4),[16,16])';
    
    target(:,:,9) = reshape(targets(3:3:end,1),[16,16])';
    target(:,:,10) = reshape(targets(3:3:end,2),[16,16])';
    target(:,:,11) = reshape(targets(3:3:end,3),[16,16])';
    target(:,:,12) = reshape(targets(3:3:end,4),[16,16])';
    
    target(find(labels(:,:,1)<1))=0;
    target(find(labels(:,:,1)<1)+16*16*1)=0;
    target(find(labels(:,:,1)<1)+16*16*2)=0;
    target(find(labels(:,:,1)<1)+16*16*3)=0;
    
    target(find(labels(:,:,2)<1)+16*16*4)=0;
    target(find(labels(:,:,2)<1)+16*16*5)=0;
    target(find(labels(:,:,2)<1)+16*16*6)=0;
    target(find(labels(:,:,2)<1)+16*16*7)=0;
    
    target(find(labels(:,:,3)<1)+16*16*8)=0;
    target(find(labels(:,:,3)<1)+16*16*9)=0;
    target(find(labels(:,:,3)<1)+16*16*10)=0;
    target(find(labels(:,:,3)<1)+16*16*11)=0;
    
    