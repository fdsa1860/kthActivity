function [label, dis, h] = findHistFew(trainCenter,X,normalize)

nr = size(trainCenter,1);
nc = size(X,1);
D = zeros(nr,nc);
thr = 0.99;
for i = 1:nr
    for j = 1:nc
        x1 = reshape(trainCenter(i,:),[],2);
        x2 = reshape(X(j,:),[],2);
        D(i,j) = hankeletAngle(x1,x2,thr);
    end
end
[dis,label] = min(D);
h = hist(label,1:nr);

if normalize
    h = h/sum(h);
end

end