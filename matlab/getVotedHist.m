% hist voting

function vh = getVotedHist(labels,d,params,normalize)

if nargin==3
    normalize = false;
end

nCluster = params.trainClusterNum{3};
prior = params.trainClusterInfo{3}.prior;
% a = params.trainClusterInfo{3}.a;
% b = params.trainClusterInfo{3}.b;
% assume k=1
% b = -1./(params.trainClusterInfo{3}.mu-1e-6);
% a = ones(size(b));
% w = gampdf(-d', a(labels), b(labels));
sig = sqrt(2)*params.trainClusterInfo{3}.sigma;
mu = zeros(size(sig));
w = 2*normpdf(-d', mu(labels), sig(labels));
w(isnan(w)) = 0;
posterior = w.*prior(labels);

vh = zeros(1,nCluster);
for i=1:length(vh)
    vh(i) = vh(i) + sum(posterior(labels==i));
end

if normalize
    vh = vh/norm(vh,1);
end

end