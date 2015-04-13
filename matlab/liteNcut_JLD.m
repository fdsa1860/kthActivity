function [label, HH_center, D]  = liteNcut_JLD(T, k, params, D)
% Ncut JLD:
% perform nCut clustering on covariance matrices with JLD metric
% Input:
% T: trajectorys, an D-by-N matrix
% k: the number of clusters
%
% Output:
% label: the clustered labeling results

N = size(T,2);
d = size(T,1);
assert(mod(d,2)==0);
nc = params.nc;
nr = (d/2-nc+1)*2;
HH = cell(1,N);
for i = 1:N
    H1 = hankel_mo(reshape(T(:,i),2,[]),[nr, nc]);
    HH1 = H1 * H1';
    HH1 = HH1 / norm(HH1,'fro');
    HH{i} = HH1 + 1e-6 * eye(size(HH1));
%       HH{i} = HH1;
end

if ~exist('D','var')
    D = zeros(N);
    for i=1:N
        for j=i:N
            if strcmp(params.metric,'JLD')
                HH1 = HH{i};
                HH2 = HH{j};
                D(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) - 0.5*log(det(HH2));
            elseif strcmp(params.metric,'binlong')
                D(j,i) = 2 - norm(HH{i}+HH{j},'fro');
            end
        end
    end
    D = D + D';
end

W = exp(-D);
NcutDiscrete = ncutW(W, k);
label = sortLabel_count(NcutDiscrete);

HH_center = cell(1, k);
for j=1:k
    if strcmp(params.metric,'JLD')
        HH_center{j} = karcher(HH{label==j});
    elseif strcmp(params.metric,'binlong')
        HH_center{j} = findCenter(HH(label==j));
    end
end

end

function center = findCenter(X)

n = length(X);
D = zeros(n,n);
for i=1:n
    for j=i+1:n
        HH1 = X{i};
        HH2 = X{j};
        D(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1*HH2));
    end
end
D = D + D';
d = sum(D);
[~,ind] = min(d);
center = X{ind};

end