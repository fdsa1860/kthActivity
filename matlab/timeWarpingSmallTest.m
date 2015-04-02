% time warping test

p1 = 0.3;
p2 = 0.5;
p3 = 0.9;
k1 = 2;
k2 = 1;
k3 = 4;

% P = [p1 p2 p3];
% Z = [z1 z2];
% sys = zpk(Z,P,1,0.1);

Ts = 0.1;
sys1 = zpk([],p1,k1,Ts);
sys2 = zpk([],p2,k2,Ts);
sys3 = zpk([],p3,k3,Ts);
sys = sys1 + sys2 + sys3;

y = impulse(sys,10);

addpath(genpath('AtomsJournal'));
N = 101; % horizon
u = 0*ones(N,1);
u(1) = 1;
ro = 1.005;
atom_arg.t_max = 150;            % No early convergence criteria checked as of now, number of steps
atom_arg.m_pole = 20;             % Number of poles to try for a gradient 
atom_arg.h_init = zeros(N,1);  % Initial value 
atom_arg.p_random = uniform_over_disc2(ro,atom_arg.t_max*atom_arg.m_pole);   % Random atoms to be tried
atom_arg.Tu = toeplitz(u,[u(1) zeros(1,N-1)]);            % Input Toeplitz matrix 
atom_arg.tau = 100;          % Tau
atom_arg.scaling = 'finite';      %  

atom_arg.ym = y-y(1);            % Output with noise
out = atomic_general_mpole(atom_arg);
h_esty2 = out.h_est;
y_est2=out.y_est;
p_est2 = out.p_est;
scale_est2 = out.scale_est;

rmpath(genpath('AtomsJournal'));


% N = 101; % horizon
% u = 0*ones(N,1);
% u(1) = 1;
% m_pole = 20;
% tao = 100;
% nBins = 50;
% 
% [h_esty, y_est,~,~,HistY] = l2_atomicBestFFT_Xikang(u,y-y(1),tao,2000,1.005,m_pole,nBins);
% 
% plot(y,'*');
% hold on; plot(y_est,'g'); hold off;
% 
% [indR,indI] = find(abs(HistY)>0.5);
% 
% p_real = ro*(2/nBins*(indR-1)-1);
% p_imag = ro*(2/nBins*(indI-1)-1);
% 
% poles = p_real + 1j*p_imag;