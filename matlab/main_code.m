% Using atoms for MIMO cases

close all
clear all

load ../expData/handwaving_1;
load ../expData/handwaving_2;
load ../expData/boxing_5;
load ../expData/boxing_6;
load ../expData/handclapping_3
load ../expData/handclapping_4

N = 100;

x(:,1) = x1(1:N) - x1(1);
x(:,2) = x2(1:N) - x2(1);
x(:,3) = x3(1:N) - x3(1);
x(:,4) = x4(1:N) - x4(1);
x(:,5) = x5(1:N) - x5(1);
x(:,6) = x6(1:N) - x6(1);

y(:,1) = y1(1:N) - y1(1);
y(:,2) = y2(1:N) - y2(1);
y(:,3) = y3(1:N) - y3(1);
y(:,4) = y4(1:N) - y4(1);
y(:,5) = y5(1:N) - y5(1);
y(:,6) = y6(1:N) - y6(1);


u = 0*ones(N,1); 
u(1) = 1;
m_pole = 20;
tao = 50 ; 
NofBins = 20;
%v = x.*x+y.*y;
v = x;
for i=1:6
[h_estx(:,i), x_est(:,i),~,~,HistX(:,:,i)]= l2_atomicBestFFT_Xikang(u, v(:,i),tao, 2000,1.005,m_pole,NofBins);
end

v = y;
for i=1:6
[h_esty(:,i), y_est(:,i),~,~,HistY(:,:,i)]= l2_atomicBestFFT_Xikang(u, v(:,i),tao, 2000,1.005,m_pole,NofBins);
end



for i=1:6
for j=1:6

    ResembleX(i,j) = trace(HistX(:,:,i)'*HistX(:,:,j))/(norm(HistX(:,:,i),'fro')^2+norm(HistX(:,:,j),'fro')^2);
    ResembleY(i,j) = trace(HistY(:,:,i)'*HistY(:,:,j))/(norm(HistY(:,:,i),'fro')^2+norm(HistY(:,:,j),'fro')^2);
    ResembleX(i,j) = 2*sum(sum(abs(HistX(:,:,i).*HistX(:,:,j))))/(sum(sum(abs(HistX(:,:,i).*HistX(:,:,i))))+sum(sum(abs(HistX(:,:,j).*HistX(:,:,j)))));
    ResembleY(i,j) = 2*sum(sum(abs(HistY(:,:,i).*HistY(:,:,j))))/(sum(sum(abs(HistY(:,:,i).*HistY(:,:,i))))+sum(sum(abs(HistY(:,:,j).*HistY(:,:,j)))));       
end
end
ResembleX
ResembleY
mesh(abs(ResembleX)+abs(ResembleY))


plot(x(:,2))
hold
plot(x_est(:,2),'r-.','LineWidth',2)
% figure 
% plot(x2)
% hold
% plot(x_est2,'r-.','LineWidth',2)
% figure
% plot(x3)
% hold
% plot(x_est3,'r-.','LineWidth',2)