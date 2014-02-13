function y = denoise(trajs, order)

addpath('../3rdParty/hstln');

y = zeros(size(trajs));
y(:,1) = trajs(:,1);
for i=1:size(trajs,1);
% for i=1:100
    xy = reshape(trajs(i,2:end),2,[]);
    mn = mean(xy,2);
    mnMat = bsxfun(@times,mn,ones(1,size(xy,2)));
    xy = xy - mnMat;
    xy_smooth = hstln_mo(xy,order);
    xy_smooth = xy_smooth + mnMat;
    y(i,2:end) = reshape(xy_smooth,1,[]);
    fprintf('hstln %d/%d traj processed.\n',i,size(trajs,1));
end

rmpath('/home/xikang/research/code/groupActivity/3rdParty/hstln');

end