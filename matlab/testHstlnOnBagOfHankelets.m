% test hstln order effect on performance of the bags of hankelets
% Xikang Zhang, 11/13/2013

% function testHstlnOnBagOfHankelets
clear;clc;close all;

params.num_km_init_word = 3;
params.MaxInteration = 3;
params.labelBatchSize = 300000;
params.actualFilterThreshold = -1;
params.find_labels_mode = 'DF';
ncenter = 300;
% svm regularization
C = 10;

addpath(genpath('../3rdParty/hankelet-master/hankelet-master'));
addpath(genpath(getProjectBaseFolder));

% Load data
file = 'action01_06_person01_26_scene01_01_20131112t.mat';
load(fullfile('../expData',file));


for order=3:6

trajs2 = denoise(trajs,order);
% trajs2 = trajs;
save(sprintf('seq_action01_06_person01_26_scene01_01_20131112d%d',order),'trajs2');
fprintf('traj2 of order %d saved.\n',order);
% save action01_06_person01_26_scene01_01_20131112d3 trajs trajs2 ofs labels counterLabels trackLabels seqLabels;
% load action01_06_person01_26_scene01_01_20131112d2;

% X = ofs(:,2:end)';
% X = trajs(:,2:end)';
X = trajs2(:,2:end)';clear trajs2;
al = reshape(al,1,[]);
pl = reshape(pl,1,[]);
sl = reshape(sl,1,[]);


% subtract mean
xm = mean(X(1:2:end,:));
ym = mean(X(2:2:end,:));
Xm = kron(ones(size(X,1)/2,1),[xm;ym]);
X = X - Xm;


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

% kmeans to learn cluster centers
rng(0); % sample
rndInd = randi(size(X_train,2),1,params.labelBatchSize);
trainCenter = cell(1, params.num_km_init_word);
for i = 1 : params.num_km_init_word
    
    assert(size(X_train,1)==30);
    [~, trainCenter{i} trainClusterMu trainClusterSigma trainClusterNum] = litekmeans_subspace(X_train(:,rndInd), ncenter,params);
    
    params.trainClusterInfo{i}.mu = trainClusterMu;
    params.trainClusterInfo{i}.sigma = trainClusterSigma;
    params.trainClusterInfo{i}.num = trainClusterNum;
    
    params.trainClusterNum{i} = size(trainCenter{i}, 2);
    
end

% labeling
params = cal_cluster_info(params);

% save kmeansWords300_action01_06_person01_26_scene01_01_20131112t params trainCenter;
save(sprintf('kmeanWords300_action01_06_person01_26_scene01_01_20131112d%d',order),'params','trainCenter');
% load kmeansWords300_action01_06_person_trainingSet_scene01_01_20131112t;

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
            [label1, dis, class_hist] = find_weight_labels_df_HHp_newProtocal({trainCenter{3}},X_tmp, params);
            hFeat = [hFeat, class_hist'];
            hsl = [hsl, usl(i)];
            hal = [hal, ual(j)];
            hpl = [hpl, upl(k)];            
        end
        fprintf('action %d/%d processed.\n',j,length(ual));
    end    
end
%     save hFeat300_action01_06_person01_26_scene01_01_20131112t hFeat hsl hal hpl;
save(sprintf('hFeat300_action01_06_person01_26_scene01_01_20131112d%d',order),'hFeat','hsl','hal','hpl');
% load hFeat300_action01_06_person01_26_scene01_01_20131112t;

X2_train = hFeat(:,ismember(hpl,trainingSet));
X2_validate = hFeat(:,ismember(hpl,validationSet));
X2_test = hFeat(:,ismember(hpl,testingSet));
y2_train = hal(:,ismember(hpl,trainingSet));
y2_validate = hal(:,ismember(hpl,validationSet));
y2_test = hal(:,ismember(hpl,testingSet));

% % normalization
% sum_X2_train = sum(X2_train,2);
% X2_train_normalized = X2_train./bsxfun(@times,sum_X2_train,ones(1,size(X2_train,2)));
% sum_X2_test = sum(X2_test,2);
% X2_test_normalized = X2_test./bsxfun(@times,sum_X2_test,ones(1,size(X2_test,2)));

% % train a SVM problem
% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% model = svmtrain(y2_train,X2_train_normalized,sprintf('-s 0 -c %d',C));
% [predict_label, accuracy, prob_estimates] = svmpredict(y2_test, X2_test_normalized, model);
% accuracy(1)
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

% train a SVM problem
addpath(genpath('../3rdParty/liblinear-1.93/matlab'));
model = train(y2_train',sparse(X2_train'),sprintf('-s 2 -c %d',C));
[predict_label, accuracy, prob_estimates] = predict(y2_test', sparse(X2_test'), model);
accuracy(1)
rmpath(genpath('../3rdParty/liblinear-1.93/matlab'));

save(sprintf('accuracy_action01_06_person01_26_scene01_01_20131112d%d',order),'accuracy');

% label_gt = [label_gt; y2_test];
% label_pred = [label_pred; predict_label];
% end
%
% disp('accuracy is ');
% nnz(label_gt-label_pred==0)/length(label_gt)


end



% 55
% end