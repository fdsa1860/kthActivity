function [score, x1_est, x2_est]  = compare_tracks(x1,x2,M,MtM,tao,tol,see_tracks)


c1  = fast_lasso(x1,0,0,M,MtM,tao,tol);

c2  = fast_lasso(x2,0,0,M,MtM,tao,tol);


e1 = c1/norm(c1);
e2 = c2/norm(c2);

score = e1'*e2;


if see_tracks
    
    x1_est = M*c1;
    x2_est = M*c2;
    
end



