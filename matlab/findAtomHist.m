function [v mainPath mainWeight] = findAtomHist(traj)

% cluster trajectories
idx = clusterTraj(traj);

% get river center paths
uIdx = unique(idx);
nCenterPath = nnz(uIdx);
centerPaths = cell(nCenterPath,1);
weights = zeros(nCenterPath,1);
counter = 1;
for i=1:length(uIdx)
    if uIdx(i)==0
        continue;
    end
    traj2 = traj(idx==uIdx(i),:);
    Rc = getRiverCenter(traj2);
    centerPaths{counter} = Rc;
    weights(counter) = nnz(idx==uIdx(i))/nnz(idx~=0);
    counter = counter + 1;
end

% set system identification paramters
N = 100; % horizon
u = 0*ones(N,1); 
u(1) = 1;
m_pole = 20;
tao = 50 ; 
nBins = 20;
v = zeros(2*nBins^2, 1);
mainPath = zeros(2*N,1);
mainWeight = 0;

% filter out short trajectories
counter = 1;
while counter <= length(centerPaths)
    if length(centerPaths{counter}) < 2*N
        centerPaths(counter) = [];
        weights(counter) = [];
        continue;
    end
    counter = counter + 1;
end

if isempty(centerPaths)
    return;
end

% % identify each center path
% W = zeros(2*nBins^2,length(centerPaths));
% for i=1:length(centerPaths)
%     track = centerPaths{i};
%     x = track(1:2:2*N);
%     y = track(2:2:2*N);
%     [h_estx, x_est,~,~,HistX]= l2_atomicBestFFT_Xikang(u, x-x(1),tao, 2000,1.005,m_pole,nBins);
%     [h_esty, y_est,~,~,HistY]= l2_atomicBestFFT_Xikang(u, y-y(1),tao, 2000,1.005,m_pole,nBins);
%     Wx = HistX(:);
%     Wy = HistY(:);
%     W(:,i) = [Wx;Wy];
% end
% 
% % return only the most probable vector
% [~,index] = max(weights);
% v = W(:,index);
% mainPath = centerPaths{index};
% mainWeight = weights(index);

[~,index] = max(weights);
track = centerPaths{index};
x = track(1:2:2*N);
y = track(2:2:2*N);
[h_estx, x_est,~,~,HistX]= l2_atomicBestFFT_Xikang(u, x-x(1),tao, 2000,1.005,m_pole,nBins);
[h_esty, y_est,~,~,HistY]= l2_atomicBestFFT_Xikang(u, y-y(1),tao, 2000,1.005,m_pole,nBins);
Wx = HistX(:);
Wy = HistY(:);
v = [Wx;Wy];
mainPath = track(1:2*N);
mainWeight = weights(index);

end