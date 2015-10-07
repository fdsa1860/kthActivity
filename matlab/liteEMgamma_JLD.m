function [label, cparams]  = liteEMgamma_JLD(T,k,params)
% EM gamma JLD:
% perform EM clustering on covariance matrices with JLD metric
% Input:
% T: trajectorys, an D-by-N matrix
% k: the number of clusters
% Output:
% label: the clustered labeling results

opt.metric = 'JLD';

N = size(T,2);
d = size(T,1);
assert(mod(d,2)==0);
nc = params.nc;
G = cell(1,N);
for i = 1:N
    H1 = hankel_mo(reshape(T(:,i),2,[]),[(d/2-nc+1)*2, nc]);
    HH1 = H1 * H1';
    HH1 = HH1 / norm(HH1,'fro');
    G{i} = HH1 + 1e-6 * eye((d/2-nc+1)*2);
%       HH{i} = HH1;
end

rng('default');
ind = randsample(length(G),k);
prior_init = 1/k;
Gm_init = zeros(size(G));
alpha_init = 1;
theta_init = 0.1;

cparams(1:k) = struct ('prior',prior_init,'Gm',Gm_init,'alpha',alpha_init,'theta',theta_init);
for i = 1:k
    cparams(i).Gm = G(ind(i));
end

[label,G_center] = kmeansJLD(G,k,opt);
D2 = HHdist(G_center, G, opt.metric);
for i = 1:k
    cparams(i).Gm = G_center(i);
    cparams(i).prior = nnz(label==i)/length(G);
    param = gamfit(D2(i,label==i));
    cparams(i).alpha = param(1);
    cparams(i).theta = param(2);
end

pre_log_lkhd = 0;
log_lkhd = 0;
log_lkhd1 = 0;
iter_max = params.MaxInteration;
iter = 1;
while abs(log_lkhd - pre_log_lkhd) >= 1e-3 * abs(log_lkhd-log_lkhd1) && iter<iter_max
    fprintf('iter %d ...\n',iter);
    pre_log_lkhd = log_lkhd;
    ez = e_step(G, k, cparams, opt);        % e step
    cparams = m_step(G, ez, cparams, opt);  % m step
    log_lkhd = lg_lkhd_gamma(G, k, cparams, opt); % log likelihood
    if iter==1
        log_lkhd1 = log_lkhd;
    end
    iter = iter + 1;
% %     if k==3     % if m=true k, plot log likelihood versus number of iteration
%         figure(15);
%         hold on;
%         plot(iter,log_lkhd,'bx');
%         xlabel('number of iterations');
%         ylabel('log likelihood');
%         title('gamma mix log likelihood');
%         hold off;
% %     end
end

ez = e_step(G, k, cparams, opt);
[~,label] = max(ez);

end