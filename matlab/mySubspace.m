function theta = mySubspace(A,B)
% mySubspace
% compute subspace based on Hendra Gunawan et al, A Formula for Angles
% between Subspaces of Inner Product Spaces

% Compute orthonormal bases, using SVD in "orth" to avoid problems
% when A and/or B is nearly rank deficient.
A = orth(A);
B = orth(B);
%Check rank and swap
if size(A,2) < size(B,2)
   tmp = A; A = B; B = tmp;
end

M = B'*A;

M_tilde = zeros(size(B,2),size(A,2));
for i = 1:size(M_tilde,1)
    for j = 1:size(M_tilde,2)
        inda = 1:size(A,2);
        tmp1 = [B(:,i) A(:,inda~=j)];
        tmp2 = [A(:,j) A(:,inda~=j)];
        tmp = tmp1'*tmp2;
        M_tilde(i,j) = det(tmp);
    end
end

theta = acos(min(1,sqrt(det(M*M_tilde'))));

end