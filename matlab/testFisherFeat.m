% test fisher vectors
% Xikang Zhang, 06/04/2014

% function testHankelFeat
clear;clc;close all;

% params.num_km_init_word = 3;
% params.MaxInteration = 3;
params.labelBatchSize = 300000;
% params.actualFilterThreshold = -1;
% params.find_labels_mode = 'DF';
ncenter = 256;
% % ncenter = 100;
% % svm regularization
% C = 10;
% G = 0.015;
% 
% addpath(genpath('../3rdParty/hankelet-master/hankelet-master'));
% addpath(genpath(getProjectBaseFolder));

% Load data
file = 'seq_action01_06_person01_26_scene01_04_20131118f.mat';
load(fullfile('../expData',file));

trajs = trajs(sl~=2,:);
al = al(sl~=2);
pl = pl(sl~=2);
sl = sl(sl~=2);

% % % for order=2:6
% % 
% % % trajs2 = denoise(trajs,order);
% % % trajs2 = trajs;
% % % save(sprintf('seq_action01_06_person01_26_scene01_01_20131112d%d',order),'trajs2');
% % % fprintf('traj2 of order %d saved.\n',order);
% % 
% % % X = ofs(:,2:end)';
X = trajs(:,2:end)';
% % % X = trajs2(:,2:end)';clear trajs2;
al = reshape(al,1,[]);
pl = reshape(pl,1,[]);
sl = reshape(sl,1,[]);


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


%% gmm estimation and get fisher vectors

% rng('default'); % sample
% rndInd = randi(size(X_train,2),1,params.labelBatchSize);
% % addpath(genpath('../3rdParty/vlfeat-0.9.18/toolbox'));
% tic;
% [means, covariances, priors] = vl_gmm(X_train(:,rndInd), ncenter);
% toc
% % rmpath(genpath('../3rdParty/vlfeat-0.9.18/toolbox'));
% save gmm300_action01_06_person01_26_scene010304_20140604t means covariances priors;
% load ../expData/gmm256_action01_06_person01_26_scene010304_20140604f;

%% get fisher vector features
% numScene = 3; 
% numAction = 6;
% numPerson = 25;
% N = numScene * numAction * numPerson;
% D = size(X,1);
% fFeat = zeros(2*ncenter*D,N);
% fsl = zeros(1,N);
% fal = zeros(1,N);
% fpl = zeros(1,N);
% counter = 1;
% indToErase = [];
% for i=[1,3,4]
%     for j=1:6
%         for k=1:25
%             X_tmp = X(:,sl==i & al==j & pl==k);
%             if isempty(X_tmp)
%                 indToErase = [indToErase counter];
%                 continue;
%             end
%             class_hist = vl_fisher(X_tmp, means, covariances, priors);
%             fFeat(:,counter) = class_hist;
%             fsl(counter) = i;
%             fal(counter) = j;
%             fpl(counter) = k;
%             counter = counter + 1;
%         end
%         fprintf('action %d/%d processed.\n',j,6);
%     end    
% end
% fFeat(:,indToErase) = [];
% fsl(indToErase) = [];
% fal(indToErase) = [];
% fpl(indToErase) = [];
%     save fFeat300_action01_06_person01_26_scene010304_20140604t fFeat fsl fal fpl;
load ../expData/fFeat256_action01_06_person01_26_scene010304_20140604f;

% % L1 normalization
% sum_fFeat = sum(abs(fFeat),1);
% fFeat = fFeat./bsxfun(@times,sum_fFeat,ones(size(fFeat,1),1));
% % power normalization
% fFeat = sign(fFeat).*abs(fFeat).^0.5;
% % L2 normalization
% sum_fFeat = sum(fFeat.^2,1).^0.5;
% fFeat = fFeat./bsxfun(@times,sum_fFeat,ones(size(fFeat,1),1));

X2_train = fFeat(:,ismember(fpl,trainingSet));
X2_validate = fFeat(:,ismember(fpl,validationSet));
X2_test = fFeat(:,ismember(fpl,testingSet));
y2_train = fal(:,ismember(fpl,trainingSet));
y2_validate = fal(:,ismember(fpl,validationSet));
y2_test = fal(:,ismember(fpl,testingSet));

% X2_train = [X2_train X2_validate];
% y2_train = [y2_train y2_validate];
% y2_train(y2_train~=3) = 1;
% y2_test(y2_test~=3) = 1;


%% train a SVM problem using one versus all, liblinear
addpath(genpath('../3rdParty/liblinear-1.93/matlab'));
Cind = -1:10;
C = 2.^Cind;
C = 100;
accuracyMat = zeros(1,length(C));
for ci = 1:length(C)
    ly = unique(y2_train);
    svmModel = cell(1,length(ly));
    accuracy = zeros(1,length(ly));
    for i=1:length(ly)
        y_train2 = y2_train;
%         y_train2(y2_train==ly(i)) = 1;
%         y_train2(y2_train~=ly(i)) = -1;
        y_validate2 = y2_validate;
%         y_validate2(y2_validate==ly(i)) = 1;
%         y_validate2(y2_validate~=ly(i)) = -1;
        y_test2 = y2_test;
%         y_test2(y2_test==ly(i))=1;
%         y_test2(y2_test~=ly(i))=-1;
        model = train(y_train2',sparse(X2_train'),sprintf('-s 2 -c %d',C(ci)));
        [predict_label, ~, prob_estimates] = predict(y_validate2', sparse(X2_validate'), model);
        accuracy(i) = nnz(predict_label==y_validate2')/length(y_validate2);
%             [predict_label, ~, prob_estimates] = predict(y_test2', sparse(X2_test'), model);
%             accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
        svmModel{i} = model;
    end
    
    % accuracy
    fprintf('\naccuracy is %f\n',mean(accuracy));
    accuracyMat(ci) = mean(accuracy);
end
rmpath(genpath('../3rdParty/liblinear-1.93/matlab'));

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
% addpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
% Cind = -10:10;
% G = 10.^Cind;
% C = 10.^Cind;
% % G = 1e-4;
% % C = 10;
% % G = 1;
% % C = 1e2;
% % gi2 = [6 7 6 7 7 7];
% % ci2 = [13 12 12 12 12 12];
% accuracyMat = zeros(length(G),length(C));
% for gi = 1:length(G)
%     for ci = 1:length(C)
% % ly = unique(y2_train);
% % svmModel = cell(1,length(ly));
% % accuracy = zeros(1,length(ly));
% % for i=1:length(ly)
% %     y_train2 = y2_train;
% %     y_train2(y2_train==ly(i)) = 1;
% %     y_train2(y2_train~=ly(i)) = -1;
% %     y_validate2 = y2_validate;
% %     y_validate2(y2_validate==ly(i)) = 1;
% %     y_validate2(y2_validate~=ly(i)) = -1;
% %     y_test2 = y2_test;
% %     y_test2(y2_test==ly(i))=1;
% %     y_test2(y2_test~=ly(i))=-1;
%     model = svmtrain_chi2(y2_train',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi),C(ci)));
% %     model = svmtrain_chi2(y_train2',X2_train',sprintf('-t 5 -g %f -c %d -q',G(gi2(i)),C(ci2(i))));
%     [predict_label, ~, prob_estimates] = svmpredict_chi2(y2_validate', X2_validate', model);
%     accuracy = nnz(predict_label==y2_validate')/length(y2_validate);
% %     [predict_label, ~, prob_estimates] = svmpredict_chi2(y_test2', X2_test', model);
% %     accuracy(i) = nnz(predict_label==y_test2')/length(y_test2);
% %     svmModel{i} = model;
% % end
% 
% 
% % accuracy
% fprintf('\naccuracy is %f\n',accuracy);
% accuracyMat(gi,ci) = accuracy;
% % accuracyMat(gi,ci,:) = accuracy;
% 
%     end
% end
% rmpath('/home/xikang/research/code/kthActivity/3rdParty/libsvm-2.9-dense_chi_square_mat');
% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');

% label_gt = [label_gt; y2_test];
% label_pred = [label_pred; predict_label];
% end
%
% disp('accuracy is ');
% nnz(label_gt-label_pred==0)/length(label_gt)
