function [u_hat,eta,x,R] = incremental_hstln_mo(u,eta_max,Omega)

[u_D,u_N] = size(u);
Rmax = u_N;

if nargin<4
    Omega = ones(1,u_N);
    

end
    
u_N_1 = sum(Omega);
    

for R=1:Rmax
    [u_hat,eta,x] = hstln_mo(u,R,'',Omega);
    
%     norm_eta = sqrt(sum(sum(eta.^2,1).*Omega));
    norm_eta = max(max(abs(eta)).*Omega);% modified by xikang
%     if norm_eta/u_N_1 < eta_max
    if norm_eta < eta_max % modified by xikang    
        break;
    end
end