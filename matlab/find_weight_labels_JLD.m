function [label, dis, histX] = find_weight_labels_JLD(HH_center,T,params)
% compute the distance for testing data samples to the trained cluster
% centers
% Input:
% HH_center:    cluster center samples
% T:            Testing samples
% params:       container for supporting information for previous functions
% Output:
% label:        label for each testing samples
% dis:          distance to cluster centers according to each data sample
% histX:        histogram of X for different cluster centers (histogram for

N = size(T,2);
d = size(T,1);
nc = params.nc;
nr = (d/2-nc+1)*2;
assert(mod(d,2)==0);
HH = cell(1,N);
for i = 1:N
    H1 = hankel_mo(reshape(T(:,i),2,[]),[nr, nc]);
    HH1 = H1 * H1';
    HH1 = HH1 / norm(HH1,'fro');
    HH{i} = HH1 + 1e-6 * eye(size(HH1));
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
for i=1:k
    for j=1:N
        HH1 = HH_center{i};
        HH2 = HH{j};
%         D(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) - 0.5*log(det(HH2));
        D(i,j) = JLD(HH1,HH2);
    end
end
[~,label] = min(D);
histX = histc(label, 1:k);
dis = [];

end
