clc;close all;clear;
%% load trajectory data
load ../expData/seq_action01_06_person01_26_scene01_04_20131118t;

%% show the sample trajectories
vind1 = find(sl==1 & pl==1 & al==1); 
vind2 = find(sl==1 & pl==2 & al==1);
vind3 = find(sl==1 & pl==1 & al==3);
vind4 = find(sl==1 & pl==2 & al==3);

trajs1 = trajs(vind1,:);
trajs2 = trajs(vind2,:);
trajs3 = trajs(vind3,:);
trajs4 = trajs(vind4,:);

%% system identification of trajectories
N = 15; % horizon
u = 0*ones(N,1);
u(1) = 1;
m_pole = 20;
tao = 50;
nBins = 20;
    
fprintf('Start processing video 1.\n');

HistX1 = zeros(nBins);
HistY1 = zeros(nBins);
for ti = 1:size(trajs1,1)
x = trajs1(ti,2:2:end)';
y = trajs1(ti,3:2:end)';
[h_estx, x_est,~,~,HistX]= l2_atomicBestFFT_Xikang(u, x-x(1),tao, 2000,1.005,m_pole,nBins);
[h_esty, y_est,~,~,HistY]= l2_atomicBestFFT_Xikang(u, y-y(1),tao, 2000,1.005,m_pole,nBins);
HistX1 = HistX1 + abs(HistX);
HistY1 = HistY1 + abs(HistY);
end

fprintf('Start processing video 2.\n');

HistX2 = zeros(nBins);
HistY2 = zeros(nBins);
for ti = 1:size(trajs2,1)
x = trajs2(ti,2:2:end)';
y = trajs2(ti,3:2:end)';
[h_estx, x_est,~,~,HistX]= l2_atomicBestFFT_Xikang(u, x-x(1),tao, 2000,1.005,m_pole,nBins);
[h_esty, y_est,~,~,HistY]= l2_atomicBestFFT_Xikang(u, y-y(1),tao, 2000,1.005,m_pole,nBins);
HistX2 = HistX2 + abs(HistX);
HistY2 = HistY2 + abs(HistY);
end

fprintf('Start processing video 3.\n');

HistX3 = zeros(nBins);
HistY3 = zeros(nBins);
for ti = 1:size(trajs3,1)
x = trajs3(ti,2:2:end)';
y = trajs3(ti,3:2:end)';
[h_estx, x_est,~,~,HistX]= l2_atomicBestFFT_Xikang(u, x-x(1),tao, 2000,1.005,m_pole,nBins);
[h_esty, y_est,~,~,HistY]= l2_atomicBestFFT_Xikang(u, y-y(1),tao, 2000,1.005,m_pole,nBins);
HistX3 = HistX3 + abs(HistX);
HistY3 = HistY3 + abs(HistY);
end

fprintf('Start processing video 4.\n');

HistX4 = zeros(nBins);
HistY4 = zeros(nBins);
for ti = 1:size(trajs4,1)
x = trajs4(ti,2:2:end)';
y = trajs4(ti,3:2:end)';
[h_estx, x_est,~,~,HistX]= l2_atomicBestFFT_Xikang(u, x-x(1),tao, 2000,1.005,m_pole,nBins);
[h_esty, y_est,~,~,HistY]= l2_atomicBestFFT_Xikang(u, y-y(1),tao, 2000,1.005,m_pole,nBins);
HistX4 = HistX4 + abs(HistX);
HistY4 = HistY4 + abs(HistY);
end

figure(1);
subplot(221);surf(HistX1/size(trajs1,1));title('histX1');
subplot(222);surf(HistX2/size(trajs2,1));title('histX2');
subplot(223);surf(HistX3/size(trajs3,1));title('histX3');
subplot(224);surf(HistX4/size(trajs4,1));title('histX4');

figure(2);
subplot(221);surf(HistY1/size(trajs1,1));title('histY1');
subplot(222);surf(HistY2/size(trajs2,1));title('histY2');
subplot(223);surf(HistY3/size(trajs3,1));title('histY3');
subplot(224);surf(HistY4/size(trajs4,1));title('histY4');
