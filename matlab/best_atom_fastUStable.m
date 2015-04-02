% Function to find the best atom given a gradient vector and a pole 
% Author : Burak Yilmaz 
% Last Update : 12.30.2013

% p = the pole for which the atom is gonna be found 
% N = length of horizon;
% is_IC : True if atom is to be used for initial condition response, false
% otherwise

function [a,p_best,my_sign] = best_atom_fastUStable(grad_f,p,N,is_IC,m)
scale = 1 - abs(p).^2;   % Scale vector (m,1)
grad_f = grad_f';
%N = length(grad_f);
%Ncomp = min(ceil(-23.0259/log(abs(p))),N);  % p^(Ncomp) ~ 1e-10;
p_norm = abs(p);
Ncomp_v = min(ceil(-34.5388./log(abs(p))),N);  % p^(Ncomp) ~ 1e-15;
Ncomp_v(Ncomp_v<0) = N;
scale(scale<=0.01) = 1;
a = zeros(N,1);
B = zeros(max(Ncomp_v),2*m);
if is_IC
    for i=1:m
        Ncomp = Ncomp_v(i);
        dummy(1:Ncomp-1,1) = p(i);
        %comp_vec = p.^(0:N-1);
        comp_vec = cumprod(dummy(1:Ncomp-1),1);
        B(1:Ncomp-1,2*i-1) = scale(i)*(real(comp_vec));
        B(1:Ncomp-1,2*i) = scale(i)*(imag(comp_vec));
        dot_products(2*i-1:2*i) = grad_f(2:Ncomp)*B(1:Ncomp-1,2*i-1:2*i);
        
    end    
        dot_products(1:2:m*2-1)=dot_products(1:2:m*2-1)+grad_f(1)*scale';
        [~ , I] = max(abs(dot_products));
        my_sign =-sign(dot_products(I)); 
        if mod(I,2)==1        
            a(1) = scale(ceil(0.5*I))*my_sign;
            a(2:Ncomp_v(ceil(0.5*I)),1) = my_sign*B(1:Ncomp_v(ceil(0.5*I))-1,I);
        else
            a(1)=0;
            a(2:Ncomp_v(ceil(0.5*I)),1) = my_sign*B(1:Ncomp_v(ceil(0.5*I))-1,I); 
        end
        
    
else
    for i=1:m
        Ncomp = Ncomp_v(i);
        dummy(1:Ncomp-2,1) = p(i);
        comp_vec = cumprod(dummy(1:Ncomp-2),1);    
        B(1:Ncomp-2,2*i-1) = scale(i)*(real(comp_vec));
        B(1:Ncomp-2,2*i) = scale(i)*(imag(comp_vec));
        dot_products(2*i-1:2*i) = grad_f(3:Ncomp)*B(1:Ncomp-2,2*i-1:2*i);
    end
        dot_products(1:2:m*2-1)=dot_products(1:2:m*2-1)+grad_f(2)*scale';
        [~ , I] = max(abs(dot_products));
        my_sign =-sign(dot_products(I)); 
        if mod(I,2)==1 
            a(1) = 0;
            a(2)=scale(ceil(0.5*I))*my_sign;
            a(3:Ncomp_v(ceil(0.5*I)),1) = my_sign*B(1:Ncomp_v(ceil(0.5*I))-2,I);
        else
            a(1)=0;
            a(2)=0;
            a(3:Ncomp_v(ceil(0.5*I)),1) = my_sign*B(1:Ncomp_v(ceil(0.5*I))-2,I); 
        end
        
    
    
end
p_best = p(ceil(0.5*I));


end