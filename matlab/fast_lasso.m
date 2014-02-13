function [c, t]  = fast_lasso(y,x_init,init,K,KtK,tao,tol)


N = size(K,2);

M1 = K;
M2 = KtK;
d = K'*y;
index = randi(N);
%index = 100;
if init==0
c = zeros(N,1);
c(index) = tao*sign(rand-0.5);
else
c = x_init;    
end

p1 = K*c;
p2 = K'*p1;
grad_f = p2 - d;
norm_ck = norm(c)^2;
for t = 1:100000
    
    [~, index] = max(abs(grad_f));
    %my_index(t) = index(1);
    my_sign = -sign(grad_f(index));
    
    u = p1  - y;
    w = (tao*my_sign)*M1(:,index) - p1;
    
    alfa_r = -(u'*w)/(w'*w+1e-6);
    
    %my_alfa(t) = alfa_r;
    alfa = max(0,min(1,alfa_r));
    c_old = c;
    c = (1-alfa)*c;
    c(index) = c(index)+alfa*tao*my_sign;
    
    %norm_ckp1 = (1-alfa)^2*norm_ck+alfa^2*tao^2 + 2*(1-alfa)*alfa*tao*my_sign*(c_old(index));
    norm_ckkp = (1-alfa)*norm_ck+tao*alfa*my_sign*c_old(index);
    %criteria = sqrt((norm_ckp1+norm_ck-2*norm_ckkp)/norm_ckp1);
    dummy1 = alfa^2*(norm_ck+tao^2-2*tao*my_sign*c_old(index));
    
    criteria = sqrt(dummy1/norm_ck);
    
    norm_ck = dummy1-norm_ck+2*norm_ckkp;
    
    
    
    if criteria<=tol
        break
    end
    
    p1 = p1 + alfa*w;
    p2 = (1-alfa)*p2;
    dummy1 = (tao*my_sign*alfa);
    dummy2 = (M2(:,index(1)));
    p2 = p2 + dummy1*dummy2;
    %p2 = p2 + (tao*my_sign*alfa)*(M2(:,index(1)));
    grad_f = p2 - d;
             
end

    
