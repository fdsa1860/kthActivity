
clc;close all;clear;
%% load trajectory data
load ../expData/seq_action01_06_person01_26_scene01_04_20131118t;

%% show the sample trajectories
vind1 = find(sl==1 & pl==1 & al==1); 
vind2 = find(sl==1 & pl==1 & al==3);

ind1 = vind1(1);
ind2 = vind1(1000);
ind3 = vind2(1);
ind4 = vind2(1200);

traj1 = trajs(ind1,:);
traj2 = trajs(ind2,:);
traj3 = trajs(ind3,:);
traj4 = trajs(ind4,:);

figure(11);
hold on;plot(trajs(vind1,2:2:end)',trajs(vind1,3:2:end)','b');hold off;
axis([0 160 0 120]);
set(gca,'YDir','reverse');
hold on;plot(trajs(ind1,2:2:end)',trajs(ind1,3:2:end)','g');hold off;
hold on;plot(trajs(ind2,2:2:end)',trajs(ind2,3:2:end)','g');hold off;
legend 'boxing 1' 'boxing 2'

figure(12);
hold on;plot(trajs(vind2,2:2:end)',trajs(vind2,3:2:end)','b');hold off;
axis([0 160 0 120]);
set(gca,'YDir','reverse');
hold on;plot(trajs(ind3,2:2:end)',trajs(ind3,3:2:end)','r');hold off;
hold on;plot(trajs(ind4,2:2:end)',trajs(ind4,3:2:end)','r');hold off;
legend 'handwaving 1' 'handwaving 2'

%% system identification of trajectories
N = 15; % horizon
u = 0*ones(N,1);
u(1) = 1;
m_pole = 20;
tao = 10;
nBins = 20;

addpath(genpath('./AtomsJournal'));
ro = 1.005;                       % Radius of unit disc to be considered
atom_arg.t_max = 150;            % No early convergence criteria checked as of now, number of steps
atom_arg.m_pole = 20;             % Number of poles to try for a gradient 
atom_arg.h_init = zeros(N,1);  % Initial value 
atom_arg.p_random = uniform_over_disc2(ro,atom_arg.t_max*atom_arg.m_pole);   % Random atoms to be tried
atom_arg.Tu = toeplitz(u,[u(1) zeros(1,N-1)]);            % Input Toeplitz matrix 
atom_arg.tau = 100;          % Tau
atom_arg.scaling = 'finite';      %  

x1 = trajs(ind1,2:2:end)';
y1 = trajs(ind1,3:2:end)';
[h_estx1, x_est1,~,~,HistX1] = l2_atomicBestFFT_Xikang(u,x1-x1(1),tao,2000,1.005,m_pole,nBins);
[h_esty1, y_est1,~,~,HistY1] = l2_atomicBestFFT_Xikang(u,y1-y1(1),tao,2000,1.005,m_pole,nBins);
x2 = trajs(ind2,2:2:end)';
y2 = trajs(ind2,3:2:end)';
[h_estx2, x_est2,~,~,HistX2] = l2_atomicBestFFT_Xikang(u,x2-x2(1),tao,2000,1.005,m_pole,nBins);
[h_esty2, y_est2,~,~,HistY2] = l2_atomicBestFFT_Xikang(u,y2-y2(1),tao,2000,1.005,m_pole,nBins);
x3 = trajs(ind3,2:2:end)';
y3 = trajs(ind3,3:2:end)';
[h_estx3, x_est3,~,~,HistX3] = l2_atomicBestFFT_Xikang(u,x3-x3(1),tao,2000,1.005,m_pole,nBins);
[h_esty3, y_est3,~,~,HistY3] = l2_atomicBestFFT_Xikang(u,y3-y3(1),tao,2000,1.005,m_pole,nBins);
x4 = trajs(ind4,2:2:end)';
y4 = trajs(ind4,3:2:end)';
[h_estx4, x_est4,~,~,HistX4] = l2_atomicBestFFT_Xikang(u,x4-x4(1),tao,2000,1.005,m_pole,nBins);
[h_esty4, y_est4,~,~,HistY4] = l2_atomicBestFFT_Xikang(u,y4-y4(1),tao,2000,1.005,m_pole,nBins);

atom_arg.ym = x1-x1(1);            % Output with noise
out = atomic_general_mpole(atom_arg);h_estx1 = out.h_est;x_est1=out.y_est;
atom_arg.ym = y1-y1(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_esty1 = out.h_est;y_est1=out.y_est;

atom_arg.ym = x2-x2(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_estx2 = out.h_est;x_est2=out.y_est;
atom_arg.ym = y1-y1(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_esty2 = out.h_est;y_est2=out.y_est;

atom_arg.ym = x3-x3(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_estx3 = out.h_est;x_est3=out.y_est;
atom_arg.ym = y3-y3(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_esty3 = out.h_est;y_est3=out.y_est;

atom_arg.ym = x4-x4(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_estx4 = out.h_est;x_est4=out.y_est;
atom_arg.ym = y4-y4(1);            % Output with noise
out = atomic_general_mpole(atom_arg); h_esty4 = out.h_est;y_est4=out.y_est;

figure(1);
axisLim = [0 20 0 20 -1 1];
subplot(221);surf(HistX1);title('histX1');
axis(axisLim);title('boxing 1: x');
subplot(222);surf(HistX2);title('histX2');
axis(axisLim);title('boxing 2: x');
subplot(223);surf(HistX3);title('histX3');
axis(axisLim);title('handwaving 1: x');
subplot(224);surf(HistX4);title('histX4');axis(axisLim);
axis(axisLim);title('handwaving 2: x');

figure(2);
subplot(221);surf(HistY1);title('histY1');axis(axisLim);
axis(axisLim);title('boxing 1: y');
subplot(222);surf(HistY2);title('histY2');axis(axisLim);
axis(axisLim);title('boxing 2: y');
subplot(223);surf(HistY3);title('histY3');axis(axisLim);
axis(axisLim);title('handwaving 1: y');
subplot(224);surf(HistY4);title('histY4');axis(axisLim);
axis(axisLim);title('handwaving 2: y');

figure(3);
hold on;
plot(x1,y1,'b');
plot(x_est1+x1(1),y_est1+y1(1),'g');
plot(x2,y2,'b');
plot(x_est2+x2(1),y_est2+y2(1),'g');
plot(x3,y3,'b');
plot(x_est3+x3(1),y_est3+y3(1),'g');
plot(x4,y4,'b');
plot(x_est4+x4(1),y_est4+y4(1),'g');
hold off;

axisLim2 = [0 15 -10 10];
figure(4);
subplot(221);plot(h_estx1);
axis(axisLim2);title('boxing 1: x');
subplot(222);plot(h_estx2);
axis(axisLim2);title('boxing 2: x');
subplot(223);plot(h_estx3);
axis(axisLim2);title('handwaving 1: x');
subplot(224);plot(h_estx4);
axis(axisLim2);title('handwaving 2: x');

figure(5);
subplot(221);plot(h_esty1);
axis(axisLim2);title('boxing 1: y');
subplot(222);plot(h_esty2);
axis(axisLim2);title('boxing 2: y');
subplot(223);plot(h_esty3);
axis(axisLim2);title('handwaving 1: y');
subplot(224);plot(h_esty4);
axis(axisLim2);title('handwaving 2: y');


%% show the atom feature
% load aFeat300_action01_06_person01_26_scene010304_20140217t_temp;
% 
% idx = find(aal==6);
% 
% % for i=1:length(idx)
% % 
% % x = aFeat(1:400,idx(i));
% % y = aFeat(401:800,idx(i));
% % 
% % xsquare = reshape(x,20,20);
% % ysquare = reshape(y,20,20);
% % xysquare(:,:,i) = [xsquare;ysquare];
% % 
% % end
% 
% % figure(6);
% % for i=1:size(xysquare,3)
% %     
% %     hold on;contour(xysquare(:,:,i));hold off;
% % end
% 
% figure(12);
% % for i=1:length(idx)
% hold on;plot(aTraj(1:2:end,idx),aTraj(2:2:end,idx));hold off;
% axis([0 160 0 120]);
% set(gca,'YDir','reverse');
% % end