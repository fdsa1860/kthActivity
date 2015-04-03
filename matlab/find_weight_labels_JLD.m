function [label, dis, histX] = find_weight_labels_JLD(HH_center,T,params)
% compute the distance for testing data samples to the trained cluster
% centers
% Input:
% center:       cluster center samples
% X:            Testing samples
% params:       container for supporting information for previous functions
% Output:
% labels:       label for each testing samples
% dis:          distance to cluster centers according to each data sample
% histX:        histogram of X for different cluster centers (histogram for

nc = params.nc;

N = size(T,2);
d = size(T,1);
assert(mod(d,2)==0);
HH = cell(1,N);
for i = 1:N
    H1 = hankel_mo(reshape(T(:,i),2,[]),[(d/2-nc+1)*2, nc]);
    H1_p = H1 / (norm(H1*H1','fro')^0.5);
    HH1 = H1_p' * H1_p;
    HH{i} = HH1 + 1e-6 * eye(nc);
%       HH{i} = HH1;
end


% k = params.num_clusterNum;
% HH_center = cell(1,k);
% for i = 1:k
%     H1 = hankel_mo(reshape(center(:,i),2,[]),[(d/2-nc+1)*2, nc]);
%     H1_p = H1 / (norm(H1*H1','fro')^0.5);
%     HH1 = H1_p' * H1_p;
%     HH_center{i} = HH1 + 1e-6 * eye(nc);
%     %       HH{i} = HH1;
% end

k = length(HH_center);
D = zeros(k,N);
for i=1:N
    for j=1:k
        HH1 = HH{i};
        HH2 = HH_center{j};
        D(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) - 0.5*log(det(HH2));
    end
end
[~,label] = min(D);
histX = histc(label, 1:k);
dis = [];

end