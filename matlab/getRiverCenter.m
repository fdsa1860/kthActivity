function Rc = getRiverCenter(traj)
% build a river center matrix

x = traj(:,2:2:end);
y = traj(:,3:2:end);
L = (size(traj,2)-1)/2;

fr = traj(:,1);
fr = fr - fr(1) + L;
sizeOfRc = fr(end) - fr(1) + L;
numOfTraj = size(traj,1);
Rx = zeros(numOfTraj,sizeOfRc);
Ry = zeros(numOfTraj,sizeOfRc);
for i=1:numOfTraj
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