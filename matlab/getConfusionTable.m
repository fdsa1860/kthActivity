function C = getConfusionTable(y, yp,normalize)

if nargin<3
    normalize = true;
end

% confusion matrix
lut = unique(y);
n = length(lut);
C = zeros(n,n);
for i=1:length(y)
    C(lut==y(i),lut==yp(i)) = C(lut==y(i),lut==yp(i)) + 1;
end

if normalize
    sum_C = sum(C,2);
    C = C./bsxfun(@times,sum_C, ones(1,n));
end

end