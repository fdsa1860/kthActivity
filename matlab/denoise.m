function Tout = denoise(T,verbose)

xm = mean(T(1:2:end,:));
ym = mean(T(2:2:end,:));
Tm = kron(ones(size(T,1)/2,1),[xm;ym]);
T = T - Tm;

Tout = zeros(size(T));
N = size(T,2);
load ind;
% for i=1:N
for i=ind
    u = reshape(T(:,i),2,[]);
    [u_hat,eta,r,R] = incremental_hstln_mo(u,0.2*norm(u));
    Tout(:,i)  = u_hat(:);
    fprintf('hstln %d/%d traj processed.\n',i,N);
    if verbose
        plot(u(1,:),u(2,:),'b.-');hold on;plot(u_hat(1,:),u_hat(2,:),'go-');hold off;
        pause;
    end
end

Tout = Tout + Tm;

end