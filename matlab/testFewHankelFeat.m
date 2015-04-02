% test the bags of hankelets with only a few trajectories
% Xikang Zhang, 05/21/2014

% function testHstlnOnBagOfHankelets
clear;clc;close all;

params.num_km_init_word = 3;
params.MaxInteration = 3;
params.labelBatchSize = 1000;
params.actualFilterThreshold = -1;
params.find_labels_mode = 'DF';
ncenter = 100;
% svm regularization
C = 10;

% % Load data
% trajPath = '/home/xikang/research/code/kthActivity/3rdParty/dense_trajectory_release_v1.2';
% file = 'person01_boxing_d1_uncomp_features2';
% trajs = importdata(fullfile(trajPath,file));

% Load data
file = 'trajs64_action01_06_person01_26_scene01_01_20140523.mat';
load(fullfile('../expData',file));
% 
% % only consider scene 1
% X = trajs(sl==1,2:end);
% al = al(sl==1);
% pl = pl(sl==1);
% sl = sl(sl==1);
% 
% tic
% addpath('../3rdParty/hstln');
% eta_thr = 1;
% N = size(X,1);
% X2 = zeros(size(X));
% for i=1:N
%     x1 = reshape(X(i,:),2,[]);
%     [x2,eta,x,R] = fast_incremental_hstln_mo(x1,eta_thr);
% %     R
%     X2(i,:) = reshape(x2,1,[]);
%     fprintf('%d / %d \n',i,N);
% end
% rmpath('../3rdParty/hstln');
% toc

% file = 'action01_06_person01_26_scene01_01_eta1_20140523.mat';
% % save action01_06_person01_26_scene01_01_eta1_20140523 X2;
% load(fullfile('../expData',file));

% X = X2; % fit the format

% divide data into training, validation and testing sets
trainingSet = [11, 12, 13, 14, 15, 16, 17, 18];
validationSet = [19, 20, 21, 23, 24, 25, 1, 4];
testingSet = [22, 2, 3, 5, 6, 7, 8, 9, 10];

% X_train = X2(ismember(pl,trainingSet),:);
% a_train = al(ismember(pl,trainingSet),:);
% s_train = sl(ismember(pl,trainingSet),:);
% p_train = pl(ismember(pl,trainingSet),:);
% X_validate = X2(ismember(pl,validationSet),:);
% a_validate = al(ismember(pl,validationSet),:);
% s_validate = sl(ismember(pl,validationSet),:);
% p_validate = pl(ismember(pl,validationSet),:);
% X_test = X2(ismember(pl,testingSet),:);
% a_test = al(ismember(pl,testingSet),:);
% s_test = sl(ismember(pl,testingSet),:);
% p_test = pl(ismember(pl,testingSet),:);

% % ncut to learn cluster centers
% rng(0); % sample
% N = size(X_train,1);
% rndInd = randsample(N, params.labelBatchSize);
% [label,trainCenter,W] = nCutContour2(X_train(rndInd,:),ncenter);

% file = 'nCut_words100_action01_06_person01_25_scene01_01_20140523';
% % save nCut_words100_action01_06_person01_25_scene01_01_20140523 trainCenter;
% load(fullfile('../expData',file));

% % get hankelet features
% numScene = 1; 
% numAction = 6;
% numPerson = 25;
% N = numScene * numAction * numPerson;
% hFeat = zeros(N,ncenter);
% hsl = zeros(N,1);
% hal = zeros(N,1);
% hpl = zeros(N,1);
% counter = 1;
% indToErase = [];
% for i=1
%     for j=1:6
%         for k=1:25
%             X_tmp = X2(sl==i & al==j & pl==k,:);
%             if isempty(X_tmp)
%                 indToErase = [indToErase counter];
%                 continue;
%             end
%             [label, dis, class_hist] = findHistFew(trainCenter,X_tmp,1);
%             hFeat(counter,:) = class_hist;
%             hsl(counter) = i;
%             hal(counter) = j;
%             hpl(counter) = k;
%             counter = counter + 1;
%         end
%         fprintf('action %d/%d processed.\n',j,6);
%     end    
% end
% hFeat(indToErase,:) = [];
% hsl(indToErase) = [];
% hal(indToErase) = [];
% hpl(indToErase) = [];
%     save hFeat100_action01_06_person01_25_scene01_01_20140523 hFeat hsl hal hpl;
file = 'hFeat100_action01_06_person01_25_scene01_01_20140523';
load(fullfile('../expData',file));

hFeat_train = hFeat(ismember(hpl,trainingSet),:);
hFeat_validate = hFeat(ismember(hpl,validationSet),:);
hFeat_test = hFeat(ismember(hpl,testingSet),:);
hl_train = hal(ismember(hpl,trainingSet),:);
hl_validate = hal(ismember(hpl,validationSet),:);
hl_test = hal(ismember(hpl,testingSet),:);

% % train a SVM problem
% addpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));
% model = svmtrain(y2_train,X2_train_normalized,sprintf('-s 0 -c %d',C));
% [predict_label, accuracy, prob_estimates] = svmpredict(y2_test, X2_test_normalized, model);
% accuracy(1)
% rmpath(genpath('/home/xikang/research/code/groupActivity/3rdParty/libsvm-3.17'));

% % train a SVM problem
% addpath(genpath('../3rdParty/liblinear-1.93/matlab'));
% model = train(y2_train,sparse(X2_train),sprintf('-s 2 -c %d',C));
% [predict_label, accuracy, prob_estimates] = predict(y2_test, sparse(X2_test), model);
% accuracy(1)
% rmpath(genpath('../3rdParty/liblinear-1.93/matlab'));
% 
% save(sprintf('accuracy_action01_06_person01_26_scene01_01_20131112d%d',order),'accuracy');

%% chi square svm
addpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
% use cross validation to decide parameters
GInd = -7:1;
CInd = -3:5;
G = 5.^GInd;
C = 5.^CInd;
% G = 1;
% C = 1e2;

N = size(hFeat_train,1);
accuracyMat = zeros(length(G),length(C));
    
    for gi = 1:length(G)
        for ci = 1:length(C)
            model = svmtrain_chi2(hl_train,hFeat_train,sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
            [predict_label, ~, ~] = svmpredict_chi2(hl_test, hFeat_test, model);
            accuracyMat(gi,ci) = nnz(predict_label==hl_test)/length(hl_test);
        end
    end

accuracyMat
[sub_a,sub_b] = find(accuracyMat==max(max(accuracyMat)));

% G = 1e-4;
% C = 100;
% G = 1;
% C = 5;
model = svmtrain_chi2(hl_train,hFeat_train,sprintf('-t 5 -g %f -c %d -q',G(sub_a(1)),C(sub_b(1))));
[predict_label, ~, prob_estimates] = svmpredict_chi2(hl_test, hFeat_test, model);
accuracy = nnz(predict_label==hl_test)/length(hl_test)
rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');

getConfusionTable(hl_test,predict_label,1)
