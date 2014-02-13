% test atom vector and kmeans
% Xikang Zhang, 01/31/2014

% function testHankelFeat
clear;clc;close all;

params.num_km_init_word = 3;
params.MaxInteration = 3;
params.labelBatchSize = 300000;
params.actualFilterThreshold = -1;
params.find_labels_mode = 'DF';
ncenter = 300;

% Load data
file = 'seq_action01_06_person01_26_scene01_04_20131118t.mat';
load(fullfile('../expData',file));


X = trajs(:,2:end)';
al = reshape(al,1,[]);
pl = reshape(pl,1,[]);
sl = reshape(sl,1,[]);

% subtract mean
xm = mean(X(1:2:end,:));
ym = mean(X(2:2:end,:));
Xm = kron(ones(size(X,1)/2,1),[xm;ym]);
X = X - Xm;

Wx = zeros(721,size(X,2));
Wy = zeros(721,size(X,2));
load M;
K = M(1:15,:);
KtK = K'*K;
tao = 100;
tol = 1e-3;
tic;
for i=1:size(X,2)
    Wx(:,i)  = fast_lasso(X(1:2:end,i),0,0,K,KtK,tao,tol);
    Wy(:,i)  = fast_lasso(X(2:2:end,i),0,0,K,KtK,tao,tol);
    if mod(i,1000)==0; display(i); end
end
toc
W = [Wx;Wy];
save temp W;

% to rectify biased data, get mirror trajectories
X_mirror = bsxfun(@times,X,kron(ones(15,1),[-1; 1]));
X = [X X_mirror];
al = [al al];
pl = [pl pl];
sl = [sl sl+length(unique(sl))];

% divide data into training, validation and testing sets
trainingSet = [11, 12, 13, 14, 15, 16, 17, 18];
validationSet = [19, 20, 21, 23, 24, 25, 1, 4];
testingSet = [22, 2, 3, 5, 6, 7, 8, 9, 10];

X_train = X(:,ismember(pl,trainingSet));
a_train = al(:,ismember(pl,trainingSet));
s_train = sl(:,ismember(pl,trainingSet));
p_train = pl(:,ismember(pl,trainingSet));
X_validate = X(:,ismember(pl,validationSet));
a_validate = al(:,ismember(pl,validationSet));
s_validate = sl(:,ismember(pl,validationSet));
p_validate = pl(:,ismember(pl,validationSet));
X_test = X(:,ismember(pl,testingSet));
a_test = al(:,ismember(pl,testingSet));
s_test = sl(:,ismember(pl,testingSet));
p_test = pl(:,ismember(pl,testingSet));


% % kmeans
% rng(0); % sample
% rndInd = randi(size(X_train,2),1,params.labelBatchSize);
% X_train_samples = X_train(:,rndInd);
% Wx = zeros(721,size(X_train_samples,2));
% Wy = zeros(721,size(X_train_samples,2));
% load M;
% K = M(1:15,:);
% KtK = K'*K;
% tao = 100;
% tol = 1e-3;
% for i=1:size(X_train_samples,2)
%     Wx(:,i)  = fast_lasso(X_train_samples(1:2:end,i),0,0,K,KtK,tao,tol);
%     Wy(:,i)  = fast_lasso(X_train_samples(2:2:end,i),0,0,K,KtK,tao,tol);
%     i
% end
% W = [Wx;Wy];
% [idx,c] = kmeans(W',ncenter);
% save kmeansAtomWords300_action01_06_person01_26_scene01_04_20140131 trainCenters;
load kmeansAtomWords300_action01_06_person01_26_scene01_04_20140131;

% get hankelet features
hFeat = [];
hsl = [];
hal = [];
hpl = [];
usl = unique(sl);
for i=1:length(usl)
    ual = unique(al(sl==usl(i)));
    for j=1:length(ual)
        upl = unique(pl(sl==usl(i) & al==ual(j)));
        for k=1:length(upl)            
            X_tmp = X(:,sl==usl(i) & al==ual(j) & pl==upl(k));
            [label1, dis, class_hist] = findAtomlabels(trainCenter,X_tmp);
%             class_hist2 = getVotedHist(label1,dis,params);
            hFeat = [hFeat, class_hist'];
            hsl = [hsl, usl(i)];
            hal = [hal, ual(j)];
            hpl = [hpl, upl(k)];            
        end
        fprintf('action %d/%d processed.\n',j,length(ual));
    end    
end
%     save hFeat300_action01_06_person01_26_scene01_04_20131118t hFeat hsl hal hpl;
load ../expData/hFeat400_action01_06_person01_26_scene01_04_20140124t;

X2_train = hFeat(:,ismember(hpl,trainingSet));
X2_validate = hFeat(:,ismember(hpl,validationSet));
X2_test = hFeat(:,ismember(hpl,testingSet));
y2_train = hal(:,ismember(hpl,trainingSet));
y2_validate = hal(:,ismember(hpl,validationSet));
y2_test = hal(:,ismember(hpl,testingSet));


Cind = -10:10;
G = 10.^Cind;
C = 10.^Cind;
% G = 1e-4;
% C = 10;
G = 1e-4;
C = 1e2;
accuracyMat = zeros(length(G),length(C));
for gi = 1:length(G)
    for ci = 1:length(C)

addpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
ly = unique(y2_train);
svmModel = cell(1,length(ly));
accuracy = zeros(1,length(ly));
for i=1:length(ly)
    y_train2 = y2_train;
    y_train2(y2_train==ly(i)) = 1;
    y_train2(y2_train~=ly(i)) = -1;
    y_validate2 = y2_validate;
    y_validate2(y2_validate==ly(i)) = 1;
    y_validate2(y2_validate~=ly(i)) = -1;
    y_test2 = y2_test;
    y_test2(y2_test==ly(i))=1;
    y_test2(y2_test~=ly(i))=-1;
    model = svmtrain_chi2(y_train2',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
    [predict_label, ~, prob_estimates] = svmpredict_chi2(y_validate2', X2_validate', model);
    accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
%     [predict_label, ~, prob_estimates] = svmpredict_chi2(y_test2', X2_test', model);
%     accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
    svmModel{i} = model;
end
rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');

% accuracy
fprintf('\naccuracy is %f\n',mean(accuracy));
accuracyMat(gi,ci) = mean(accuracy);

    end
end
% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');



% 55
% end