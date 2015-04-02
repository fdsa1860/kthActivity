% Data
function p = uniform_over_disc2(rho,n)


theta = rand(1,n)*(2*pi);
r = sqrt(rand(1,n))*rho;
p = r.*cos(theta)+1j*r.*sin(theta);

end



% Engine
