% function testHankelFeat
clear;clc;close all;

% svm regularization parameters
C = 10;
G = 1e-4;

addpath(genpath('../3rdParty/hankelet-master/hankelet-master'));
addpath(genpath(getProjectBaseFolder));

% load data
trajPath = '../expData';
fileName = 'xikang_handwaving';
traj = load(fullfile(trajPath,fileName));

% ignore frame number
X = traj(:,2:end)';

% take out the mean
xm = mean(X(1:2:end,:));
ym = mean(X(2:2:end,:));
Xm = kron(ones(size(X,1)/2,1),[xm;ym]);
X = X - Xm;

x = traj(:,2:2:end);
y = traj(:,3:2:end);
xt = X(1:2:end,:);
yt = X(2:2:end,:);
xy = [x,y];
traj2 = traj; traj2(:,1) = traj2(:,1)*0.1;
z = linkage(traj2);
numOfCluster = 5;
ColorSet = 'bgrymck';
idx = cluster(z,'maxclust',numOfCluster);
% plot
figure;
set(gca,'YDir','reverse');
hold on;
for i=1:numOfCluster
    plot(x(idx==i,:)',y(idx==i,:)',ColorSet(1+mod(i,length(ColorSet))));
end
hold off;

% build a matrix
fr = traj(:,1);
lastFr = fr(end);
Rx = zeros(size(traj,1),lastFr);
Ry = zeros(size(traj,1),lastFr);
for i=1:size(traj,1)
    if idx(i)~=2, continue; end
    Rx(i,fr(i)-14:fr(i)) = x(i,:);
    Ry(i,fr(i)-14:fr(i)) = y(i,:);
end
xBaseNum = sum(Rx~=0);
yBaseNum = sum(Ry~=0);
assert(nnz(xBaseNum-yBaseNum)==0);
Rxc = sum(Rx)./xBaseNum;
Ryc = sum(Ry)./yBaseNum;
figure;plot(Rxc,Ryc);
Rxm = Rxc - mean(Rxc);
Rym = Ryc - mean(Ryc);
order = 100;
Hx = hankel(Rxm(10:order),Rxm(order:end-10));
Hy = hankel(Rym(10:order),Rym(order:end-10));
H = [Hx;Hy];
figure;plot(svd(H),'*');

