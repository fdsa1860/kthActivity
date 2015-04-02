function [label,X_center,W] = nCutContour2(X,k)
% Input:
% X: an N-by-d matrix
% k: the number of clusters
% Output:
% label: the clustered labeling results

N = size(X,1);
thr = 0.99;

D = zeros(N,N);
for i=1:N
    for j=i+1:N
        x1 = reshape(X(i,:),[],2);
        x2 = reshape(X(j,:),[],2);
        D(i,j) = hankeletAngle(x1,x2,thr);
    end
end

D = D + D';
W = exp(-D);

addpath('../3rdParty/Ncut_9');

map = ncutW(W,k);
label = map * (1:k)';
centerInd = findCenter(W,label);
X_center = X(centerInd,:);

rmpath('../3rdParty/Ncut_9');

end

function centerInd = findCenter(W,label)

k = length(unique(label));
centerInd = zeros(k,1);
for i=1:k
    index = find(label==i);
    WW = W(index,index);
    w = sum(WW);
    [~,ind] = max(w);
    centerInd(i) = index(ind);
end

end