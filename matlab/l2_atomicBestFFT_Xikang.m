% The algorithm to solve most generic MIMO + IC case, 
% with alternating directions approach (Updating one channel at a time). 
% Author : Burak Yilmaz 
% Last Update : 12.30.2013 

% INPUTS: 

% T(:,:,i) = Toeplitz matrices of input channel i. 
% y = [y1 y2 ....]   yi is the column vector of output i. 
% t_max = maximum iteration number 
% ro = stability radius for system poles (<=1)

function [h, y_est,real_bin,imag_bin,Hist]= l2_atomicBestFFT_Xikang(u, y,tao, t_max,ro,m,NofBins)

p = uniform_over_disc2(ro,t_max*m);
N = size(y,1);
% N_fft = 2*N;
N_fft = 2^nextpow2(N);
Uf = fft([u(:);zeros(N_fft-N,1)]);
UfC = conj(Uf);
Hf = zeros(N_fft,1);
P2 = zeros(N_fft,1);
Yf = fft(y,N_fft);
h_bin = zeros(N_fft,1);
% Calculate Tu'*y
Tuy = ifft(UfC.*Yf);
h_update = zeros(N_fft,1);
%p2_bin = zeros(N_fft,1);
%cost = zeros(t_max,1);
%p2_bin = ifft(P2);
%fftw('planner', 'patient')
real_bin = zeros(t_max,1);
imag_bin = zeros(t_max,1);

Hist = zeros(NofBins,NofBins);
width = 2*ro/(NofBins);
counter_r = 1;
counter_i = 1;
for t = 1:t_max
        
         P1 = Hf.*Uf;   % Tu*h in FFT Domain
         P2 = conj(UfC.*P1)/N_fft;  % Tu'*Tu*h in FFT Domain   
         %cost(t) = norm(res_bin(1:N)-y)^2/2;
         
         p2_bin = fft(P2);
         p2_time = real(p2_bin(1:N_fft)-Tuy(1:N_fft));
         [ah,p_best,my_sign] = best_atom_fastUStable(p2_time,p((t-1)*m+1:t*m)',N,false,m);
          
         indR = floor(real(p_best)/width)+NofBins/2+1;
         indI = floor(imag(p_best)/width)+NofBins/2+1;
          
         
     
         h_bin = ifft(Hf);
         
         
         h_update(1:N) = tao*ah-h_bin(1:N);
         Hf_update = fft(h_update);
         WhF = Hf_update.*Uf;
         %alfa_r = -((res_bin(1:N)-y)'*wh_bin(1:N))/(norm(wh_bin(1:N))^2);
         alfa_r = real(-(P1-Yf)'*WhF/norm(WhF)^2);
         alfa = max(0,min(1,alfa_r));
         
         Hist(indR,indI) = Hist(indR,indI)+my_sign*alfa;
%          Hist(indR,indI) = Hist(indR,indI)+alfa;
         
         Hf = Hf + alfa*Hf_update;         
         
         

end

y_est = ifft(P1);
h = real(h_bin(1:N));
y_est = real(y_est(1:N));
           
