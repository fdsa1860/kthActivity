function idx = clusterTraj(traj, verbose)

if nargin < 2
    verbose = false;
end

numOfClusterS = 5;
thr = 0.05;

traj = sortrows(traj,1); % sort traj according to frame number

tDistThr = floor(size(traj,2)/2);
fr = traj(:,1);
idxT = ones(length(fr),1);
idxS = zeros(length(fr),1);
for i=2:length(fr)
    if fr(i) - fr(i-1) <= tDistThr
        continue;
    end
    idxT(i:end) = idxT(i:end) + 1;
end

% nearest neighbor distance with cluster tree
uidxT = unique(idxT);
counter = 1;
for i=1:length(uidxT)
    if nnz(idxT == uidxT(i)) == 1
        idxS(counter) = 1;
        counter = counter + 1;
        continue;
    end
    z = linkage(traj(idxT == uidxT(i),2:end),'centroid');
    idxS(counter:counter+size(z,1)) = cluster(z,'maxclust',numOfClusterS);
    counter = counter + size(z,1);
end

idx = (idxT-1) * numOfClusterS + idxS;

% % N-cut
% addpath('/home/xikang/research/code/kthActivity/3rdParty/Ncut_9');
% idxVector = 1:numOfCluster;
% X = traj(:,2:end);
% tic;[W,Dist] = compute_relation(X');toc
% tic;[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,numOfCluster);toc
% idx = NcutDiscrete*idxVector';
% rmpath('/home/xikang/research/code/kthActivity/3rdParty/Ncut_9');

if verbose
    x = traj(:,2:2:end);
    y = traj(:,3:2:end);
    ColorSet = 'bgrymck';
    figure;
    set(gca,'YDir','reverse');
    hold on;
    for i=1:length(unique(idx))
         plot(x(idx==i,:)',y(idx==i,:)',ColorSet(1+mod(i,length(ColorSet))));
%          pause;
    end
    hold off;
end

% filter
uidx = unique(idx);
for i=1:length(uidx)
    % filter out the outliers
    if nnz(idx==uidx(i))/length(idx) < thr
        idx(idx==uidx(i)) = 0;
    end
    % filter out temporal broken trajectories
    temporalIdx = diff(traj(idx==uidx(i),1));
    if any(temporalIdx > tDistThr)
        idx(idx==uidx(i)) = 0;
    end
end

end