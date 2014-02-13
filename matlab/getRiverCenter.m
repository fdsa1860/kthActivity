function Rc = getRiverCenter(traj)
% build a river center matrix

x = traj(:,2:2:end);
y = traj(:,3:2:end);
L = (size(traj,2)-1)/2;

fr = traj(:,1);
lastFr = fr(end);
Rx = zeros(size(traj,1),lastFr);
Ry = zeros(size(traj,1),lastFr);
for i=1:size(traj,1)
    Rx(i,fr(i)-L+1:fr(i)) = x(i,:);
    Ry(i,fr(i)-L+1:fr(i)) = y(i,:);
end
xBaseNum = sum(Rx~=0);
yBaseNum = sum(Ry~=0);
assert(nnz(xBaseNum-yBaseNum)==0);
Rxc = sum(Rx)./xBaseNum;
Ryc = sum(Ry)./yBaseNum;

Rc = [Rxc;Ryc];
Rc = Rc(:);

end