function [label, HH_center, nEachCluster]  = litekmeans_JLD(T,k,params)
% kmeansJLD:
% perform kmeans clustering on covariance matrices with JLD metric
% Input:
% T: trajectorys, an D-by-N matrix
% k: the number of clusters
% Output:
% label: the clustered labeling results

N = size(T,2);
d = size(T,1);
assert(mod(d,2)==0);
nc = params.nc;
HH = cell(1,N);
for i = 1:N
    H1 = hankel_mo(reshape(T(:,i),2,[]),[(d/2-nc+1)*2, nc]);
    H1_p = H1 / (norm(H1'*H1,'fro')^0.5);
    HH1 = H1_p * H1_p';
    HH{i} = HH1 + 1e-6 * eye((d/2-nc+1)*2);
%       HH{i} = HH1;
end

D = zeros(k,N);
ind = randsample(N,k);
HH_center = HH(ind);
label = ones(1,N);
label_old = zeros(1,N);

iter_max = params.MaxInteration;
iter = 0;
while iter<iter_max && any(label~=label_old)
    
    for i=1:N
        for j=1:k
            HH1 = HH{i};
            HH2 = HH_center{j};
            D(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) - 0.5*log(det(HH2));
        end
    end
    label_old = label;
    [~,label] = min(D);
    for j=1:k
        HH_center{j} = karcher(HH{label==j});
%         HH_center{j} = findCenter(HH(label==j));
    end
    iter = iter + 1;
    fprintf('iter %d ...\n',iter);
end

if iter==iter_max
    warning('kmeans has reached maximum iterations before converging.\n');
end

nEachCluster = histc(label, 1 : k);
[nEachCluster, IX] = sort(nEachCluster, 'descend');
HH_center = HH_center(IX);

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