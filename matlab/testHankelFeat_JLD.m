% test hankel and kmeans with JLD metric
% Xikang Zhang, 03/31/2015

% function testHankelFeat
clear;clc;close all;

params.num_km_init_word = 1;
params.MaxInteration = 100;
params.labelBatchSize = 10000;
params.actualFilterThreshold = -1;
params.find_labels_mode = 'DF';
params.nc = 8;
params.num_clusterNum = 300;
% params.num_clusterNum = 100;
% svm regularization
C = 10;
G = 0.015;

addpath(genpath('../3rdParty/hankelet-master/hankelet-master'));
addpath(genpath(getProjectBaseFolder));

% Load data
file = 'seq_action01_06_person01_26_scene01_04_20131118t.mat';
load(fullfile('../expData',file));

% trajs = trajs(sl~=2,:);
% al = al(sl~=2);
% pl = pl(sl~=2);
% sl = sl(sl~=2);

% % % for order=2:6
% % 
% % % trajs2 = denoise(trajs,order);
% % % trajs2 = trajs;
% % % save(sprintf('seq_action01_06_person01_26_scene01_01_20131112d%d',order),'trajs2');
% % % fprintf('traj2 of order %d saved.\n',order);

% X = ofs(:,2:end)';
T = trajs(:,2:end)';
% % % X = trajs2(:,2:end)';clear trajs2;
al = reshape(al,1,[]);
pl = reshape(pl,1,[]);
sl = reshape(sl,1,[]);

% average filtering and get velocity
h = [1; 1; 1]/3;
Tx = T(1:2:end,:); Tx = conv2(Tx, h, 'valid'); Vx = diff(Tx);
Ty = T(2:2:end,:); Ty = conv2(Ty, h, 'valid'); Vy = diff(Ty);
V = zeros(size(Vx,1)*2,size(Vx,2)); V(1:2:end,:) = Vx; V(2:2:end,:) = Vy;
Vnorm = sum(sqrt(Vx.^2+Vy.^2));
X = bsxfun(@rdivide,V,Vnorm);

% % subtract mean
% xm = mean(X(1:2:end,:));
% ym = mean(X(2:2:end,:));
% Xm = kron(ones(size(X,1)/2,1),[xm;ym]);
% X = X - Xm;

% % to rectify biased data, get mirror trajectories
% X_mirror = bsxfun(@times,X,kron(ones(15,1),[-1; 1]));
% X = [X X_mirror];
% al = [al al];
% pl = [pl pl];
% sl = [sl sl+4];

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


% % kmeans to learn cluster centers
% rng(0); % sample
% rndInd = randi(size(X_train,2),1,params.labelBatchSize);
% trainCenter = cell(1, params.num_km_init_word);
% for i = 1 : params.num_km_init_word
%     
%     [~, trainCenter{i}, trainClusterNum] = litekmeans_JLD(X_train(:,rndInd), params.num_clusterNum,params);
%     params.trainClusterInfo{i}.num = trainClusterNum;
%     
%     params.trainClusterNum{i} = size(trainCenter{i}, 2);
%     
% end

% % labeling
% params = cal_cluster_info(params);

% save kmeansWords300_action01_06_person01_26_scene01_04_20131118t params trainCenter;
% load ../expData/kmeansWords300_action01_06_person01_26_scene010304_20140220t_nomirror;
load ../expData/kmeansJLD_w300_action01_06_person01_26_scene01_04_20150402v;

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
            [label1, dis, class_hist] = find_weight_labels_JLD(trainCenter{1}, X_tmp, params);
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
% load ../expData/hFeat300_action01_06_person01_26_scene01_04_20131210t;
load ../expData/hFeatJLD_w300_train10000_action01_06_person01_26_scene01_04_20150401v;

X2_train = hFeat(:,ismember(hpl,trainingSet));
X2_validate = hFeat(:,ismember(hpl,validationSet));
X2_test = hFeat(:,ismember(hpl,testingSet));
y2_train = hal(:,ismember(hpl,trainingSet));
y2_validate = hal(:,ismember(hpl,validationSet));
y2_test = hal(:,ismember(hpl,testingSet));

% X2_train = [X2_train X2_validate];
% y2_train = [y2_train y2_validate];
% y2_train(y2_train~=3) = 1;
% y2_test(y2_test~=3) = 1;

% % normalization
% sum_X2_train = sum(X2_train,1);
% X2_train = X2_train./bsxfun(@times,sum_X2_train,ones(size(X2_train,1),1));
% sum_X2_validate = sum(X2_validate,1);
% X2_validate = X2_validate./bsxfun(@times,sum_X2_validate,ones(size(X2_validate,1),1));
% sum_X2_test = sum(X2_test,1);
% X2_test = X2_test./bsxfun(@times,sum_X2_test,ones(size(X2_test,1),1));

% %% train a SVM problem
% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% Cind = -10:10;
% G = 10.^Cind;
% C = 2.^Cind;
% % C = 512;
% accuracyMat = zeros(length(G),length(C));
% for gi = 1:length(G)
% for ci = 1:length(C)
%     ly = unique(y2_train);
%     svmModel = cell(1,length(ly));
%     accuracy = zeros(1,length(ly));
%     for i=1:length(ly)
%         y_train2 = y2_train;
%         y_train2(y2_train==ly(i)) = 1;
%         y_train2(y2_train~=ly(i)) = -1;
%         y_validate2 = y2_validate;
%         y_validate2(y2_validate==ly(i)) = 1;
%         y_validate2(y2_validate~=ly(i)) = -1;
%         y_test2 = y2_test;
%         y_test2(y2_test==ly(i))=1;
%         y_test2(y2_test~=ly(i))=-1;
%         model = svmtrain(y_train2',X2_train',sprintf('-s 0 -t 2 -c %d -g %d',C(ci),G(gi)));
%         [predict_label, ~, prob_estimates] = svmpredict(y_validate2', X2_validate', model);
%         accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
% %         [predict_label, ~, prob_estimates] = svmpredict(y2_test', X2_test', model);
% %         accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
%         svmModel{i} = model;
%     end
%     % accuracy
%     fprintf('\naccuracy is %f\n',mean(accuracy));
%     accuracyMat(ci,gi) = mean(accuracy);
% end
% end
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

% %% train a SVM problem using one versus all, liblinear
% Cind = -1:10;
% C = 2.^Cind;
% % C = 512;
% accuracyMat = zeros(length(G),length(C));
% for ci = 1:length(C)
%     addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/liblinear-1.93/matlab'));
%     ly = unique(y2_train);
%     svmModel = cell(1,length(ly));
%     accuracy = zeros(1,length(ly));
%     for i=1:length(ly)
%         y_train2 = y2_train;
%         y_train2(y2_train==ly(i)) = 1;
%         y_train2(y2_train~=ly(i)) = -1;
%         y_validate2 = y2_validate;
%         y_validate2(y2_validate==ly(i)) = 1;
%         y_validate2(y2_validate~=ly(i)) = -1;
%         y_test2 = y2_test;
%         y_test2(y2_test==ly(i))=1;
%         y_test2(y2_test~=ly(i))=-1;
%         model = train(y_train2',sparse(X2_train'),sprintf('-s 2 -c %d',C(ci)));
% %         [predict_label, ~, prob_estimates] = predict(y_validate2', sparse(X2_validate'), model);
% %         accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
%             [predict_label, ~, prob_estimates] = predict(y_test2', sparse(X2_test'), model);
%             accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
%         svmModel{i} = model;
%     end
%     rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/liblinear-1.93/matlab'));
%     % accuracy
%     fprintf('\naccuracy is %f\n',mean(accuracy));
%     accuracyMat(ci) = mean(accuracy);
% end

% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% ly = unique(y2_train);
% svmModel = cell(1,length(ly));
% for i=1:length(ly)
%     y_train2 = y2_train;
%     y_train2(y2_train~=ly(i))=0;
%     y_test2 = y2_test;
%     y_test2(y2_test~=ly(i))=0;
%     model = svmtrain(y_train2',X2_train',sprintf('-s 0 -t 0 -c %d',C));
%     [predict_label, accuracy, prob_estimates] = svmpredict(y_test2', X2_test', model);
%     accuracy(1)
%     svmModel{i} = model;
% end
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

% % scale data
% addpath(genpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-3.17'));
% libsvmwrite('feat_train',y2_train',sparse(X2_train'));
% libsvmwrite('feat_test',y2_test',sparse(X2_test'));
% libsvmwrite('feat_validate',y2_validate',sparse(X2_validate'));
% system('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17/svm-scale -l 0 -u 1 -s range feat_train > feat_train_scale');
% system('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17/svm-scale -l 0 -u 1 -r range feat_test > feat_test_scale');
% system('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17/svm-scale -l 0 -u 1 -r range feat_validate > feat_validate_scale');
% [y2_train_scale, X2_train_scale] = libsvmread('feat_train_scale');
% [y2_test_scale, X2_test_scale] = libsvmread('feat_test_scale');
% [y2_validate_scale, X2_validate_scale] = libsvmread('feat_validate_scale');
% rmpath(genpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-3.17'));
% X2_train = full(X2_train_scale)';
% y2_train = y2_train_scale';
% X2_test = full(X2_test_scale)';
% y2_test = y2_test_scale';
% X2_validate = full(X2_validate_scale)';
% y2_validate = y2_validate_scale';

%%
Cind = -10:10;
G = 10.^Cind;
C = 10.^Cind;
% G = 1e-4;
% C = 10;
% G = 1;
% C = 1e2;
% gi2 = [6 7 6 7 7 7];
% ci2 = [13 12 12 12 12 12];
% gi = 7;
% ci = 12;
accuracyMat = zeros(length(G),length(C),6);
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
%     model = svmtrain_chi2(y_train2',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi2(i)),C(ci2(i))));
%     [predict_label, ~, prob_estimates] = svmpredict_chi2(y_validate2', X2_validate', model);
%     accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
    [predict_label, ~, prob_estimates] = svmpredict_chi2(y_test2', X2_test', model);
    accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
    svmModel{i} = model;
end
rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');

% accuracy
fprintf('\naccuracy is %f\n',mean(accuracy));
accuracyMat(gi,ci) = mean(accuracy);
% accuracyMat(gi,ci,:) = accuracy;

    end
end
% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');

% label_gt = [label_gt; y2_test];
% label_pred = [label_pred; predict_label];
% end
%
% disp('accuracy is ');
% nnz(label_gt-label_pred==0)/length(label_gt)


% end



% 55
% end