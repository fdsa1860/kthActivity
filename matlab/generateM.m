function M = generateM

Horizon = 1000;
nTheta = 30;
nR = 1;

n = 1:Horizon;
r = linspace(1, 1, nR);
theta = linspace(0,2*pi,nTheta+1);
theta(end) = [];
unit_circle = exp(1i*theta);
cplx_p = r' * unit_circle;
cplx_p = cplx_p(:);
cplx_p = [0; 1; cplx_p];

P = bsxfun(@power, cplx_p', n');

Mr = real(P);
Mi = imag(P);

M = [Mr Mi];

end