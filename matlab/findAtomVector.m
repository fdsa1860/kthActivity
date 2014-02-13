function [v mainPath mainWeight] = findAtomVector(traj)

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

% identify each center path
load M_xik;
M_row = size(M_xik,1);
M_col = size(M_xik,2);
tao = 100;
tol = 1e-3;
W = zeros(2*M_col,length(centerPaths));

for i=1:length(centerPaths)
    track = centerPaths{i};
    L = min(length(track)/2, M_row);
%     u = ones(L,1);
%     Tu=toeplitz(u,[u(1) zeros(1,L-1)]); 
%     K = Tu*M_xik(1:L,:);
    K = M_xik(1:L,:);
    KtK = K'*K;
    x = track(1:2:2*L);
    y = track(2:2:2*L);
    Wx = fast_lasso(x-x(1),0,0,K,KtK,tao,tol);
    Wy = fast_lasso(y-y(1),0,0,K,KtK,tao,tol);
    W(:,i) = [Wx;Wy];
end

[~,index] = max(weights);
v = W(:,index);
mainPath = centerPaths{index};
mainWeight = weights(index);

end