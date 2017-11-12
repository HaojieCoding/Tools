function [t] = histpred(pred, up)


    [s,d] = hist(reshape(pred,1,224*224),10);
    summ = 0;
    for i = 1:10
        summ = summ + s(i);
        if summ/224.0/224.0 > up
            t = ceil(d(i)*10);
            break;
        end
    end