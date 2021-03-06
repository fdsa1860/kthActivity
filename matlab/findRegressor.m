function [v mainPath mainWeight] = findRegressor(traj)

% cluster trajectories
tScale = 0.1;
idx = clusterTraj(traj, tScale);

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
order = 3;
W = zeros(2*order,length(centerPaths));
for i=1:length(centerPaths)
    track = centerPaths{i};
    L = length(track)/2;
    x = track(1:2:2*L);
    y = track(2:2:2*L);
    xc = x - mean(x);
    yc = y - mean(y);
    Hx = hankel(xc(1:order),xc(order:end));
    Hy = hankel(yc(1:order),yc(order:end));
    [Ux,Sx,Vx] = svd(Hx);
    [Uy,Sy,Vy] = svd(Hy);
    rx = Ux(:,end);
    ry = Uy(:,end);
    rx = rx/rx(end);
    ry = ry/ry(end);
    W(:,i) = [rx;ry];
end

[~,index] = max(weights);
v = W(:,index);
mainPath = centerPaths{index};
mainWeight = weights(index);

end