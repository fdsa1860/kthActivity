% unitTestJLD

addpath('../../mex');

A = randn(100);
B = randn(100);
A = A*A';
B = B*B';
C = cell(1,3); C{1} = A; C{2} = A; C{3} = B;
D = cell(1,3); D{1} = A; D{2} = B; D{3} = B;

tic
for i=1:1000
d1 = log(det(0.5*A+0.5*B+1e-6*eye(size(A))))-0.5*log(det(A+1e-6*eye(size(A))))-0.5*log(det(B+1e-6*eye(size(A))));
end
toc
tic
for i=1:1000,d2 = JLD(A,B);end
toc

% d3 = JLDbatch(C,D);
% 
d1 - d2
% d1 - d3