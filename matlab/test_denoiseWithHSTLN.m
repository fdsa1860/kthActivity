% hstln denoise

eta_max = 0.01;

x1 = X(:,1321868);
u = reshape(x1,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx1 = hankel_mo(u_hat,[6 8]);HHx1=Hx1'*Hx1;HHx1=HHx1/norm(HHx1,'fro');
x2 = X(:,1469624);
u = reshape(x2,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx2 = hankel_mo(u_hat,[6 8]);HHx2=Hx2'*Hx2;HHx2=HHx2/norm(HHx2,'fro');
x3 = X(:,206033);
u = reshape(x3,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx3 = hankel_mo(u_hat,[6 8]);HHx3=Hx3'*Hx3;HHx3=HHx3/norm(HHx3,'fro');
x4 = X(:,1481929);
u = reshape(x4,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx4 = hankel_mo(u_hat,[6 8]);HHx4=Hx4'*Hx4;HHx4=HHx4/norm(HHx4,'fro');
x5 = X(:,1025987);
u = reshape(x5,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx5 = hankel_mo(u_hat,[6 8]);HHx5=Hx5'*Hx5;HHx5=HHx5/norm(HHx5,'fro');
x6 = X(:,158257);
u = reshape(x6,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx6 = hankel_mo(u_hat,[6 8]);HHx6=Hx6'*Hx6;HHx6=HHx6/norm(HHx6,'fro');
x7 = X(:,451857);
u = reshape(x7,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx7 = hankel_mo(u_hat,[6 8]);HHx7=Hx7'*Hx7;HHx7=HHx7/norm(HHx7,'fro');
x8 = X(:,887302);
u = reshape(x8,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx8 = hankel_mo(u_hat,[6 8]);HHx8=Hx8'*Hx8;HHx8=HHx8/norm(HHx8,'fro');
x9 = X(:,1553530);
u = reshape(x9,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx9 = hankel_mo(u_hat,[6 8]);HHx9=Hx9'*Hx9;HHx9=HHx9/norm(HHx9,'fro');
x10 = X(:,1565507);
u = reshape(x10,2,[]);[u_hat,eta,r,R] = incremental_hstln_mo(u, eta_max);
Hx10 = hankel_mo(u_hat,[6 8]);HHx10=Hx10'*Hx10;HHx10=HHx10/norm(HHx10,'fro');