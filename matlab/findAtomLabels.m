function [labels, dis, histX] = findAtomLabels(center, X)
% compute the distance for testing data samples to the trained cluster
% centers
% Input:
% center:       cluster center samples
% X:            Testing samples
% Output:
% labels:       label for each testing samples
% dis:          distance to cluster centers according to each data sample
% histX:        histogram of X for different cluster centers (histogram for

% Xikang Zhang
% Feb 3, 2014

ncenter = size(center,2);
histX = zeros(1, ncenter);

    
    [X_blkInfo.XHHp X_blkInfo.XHHpFrob] = calHHp(X_blk);
    
    [labels dis score]= findaword_HHp_newProtocal(centerInfo{1}, ...
        X_blkInfo, params);
    
    histX = histX + histc(labels, (1 : size(center{1}, 2)));
    
    nr_calculated  = nr_calculated + nr_tocalc;


end