function H = hankel_mo(L,nrnc)


%   H = hankel_mo(L);
%
%   H = hankel_mo(L,M);
%
%   L: Data DIMS x N 
%   M: Number of observations per row
%
% treat every column as a observation
% form a multi output hankel e.g.
%       | c1 c2 c3 |
% H =   | c2 c3 c4 |
%       | c3 c4 c5 |


[dim N] = size(L);

if nargin<2
    % nr = ceil(N/2)*dim;
    nr = ceil(N/(dim+1))*dim;
    nc = N - ceil(N/(dim+1))+1;
else
    nr = nrnc(1);
    nc = nrnc(2);
end


% nc = N - ceil(N/(dim+1))+1;
% nc = N - (nr/dim)+1;



cidx = [0 : nc-1 ];
ridx = [1 : nr]';

H = ridx(:,ones(nc,1)) + dim*cidx(ones(nr,1),:);  % Hankel subscripts 
t = L(:);

temp.type = '()';
temp.subs = {H(:)};
H = reshape( subsref( t, temp ), size( H ) );