function [Hist, bScalar] = findBriefHist(X,patt,normalize)

bFeatSize = size(patt,1);
sampleSize = size(X,2);
bFeat = zeros(bFeatSize,sampleSize);
for i = 1:bFeatSize
    bFeat(i,:) = X(patt(i,1),:) < X(patt(i,2),:);
end

bScalar = zeros(1,sampleSize);
for i = 1:bFeatSize
    bScalar = bScalar + bFeat(i,:)*2^(bFeatSize-i);
end

Hist = hist(bScalar, linspace(0,2^bFeatSize-1,256));

if normalize
    Hist = Hist/sum(Hist);
end

end