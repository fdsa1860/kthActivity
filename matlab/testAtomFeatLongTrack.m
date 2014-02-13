% test atom vector and kmeans
% Xikang Zhang, 01/31/2014

% function testAtomFeatLongTrack
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

trajs = trajs(sl~=2,:);
al = al(sl~=2);
pl = pl(sl~=2);
sl = sl(sl~=2);

X = trajs(:,2:end)';
al = reshape(al,1,[]);
pl = reshape(pl,1,[]);
sl = reshape(sl,1,[]);

% subtract mean
xm = mean(X(1:2:end,:));
ym = mean(X(2:2:end,:));
Xm = kron(ones(size(X,1)/2,1),[xm;ym]);
X = X - Xm;

% % to rectify biased data, get mirror trajectories
% X_mirror = bsxfun(@times,X,kron(ones(15,1),[-1; 1]));
% X = [X X_mirror];
% al = [al al];
% pl = [pl pl];
% sl = [sl sl+length(unique(sl))];

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


% get hankelet features
aFeat = [];
asl = [];
aal = [];
apl = [];
usl = unique(sl);
for i=1:length(usl)
    ual = unique(al(sl==usl(i)));
    for j=1:length(ual)
        upl = unique(pl(sl==usl(i) & al==ual(j)));
        for k=1:length(upl)            
            X_tmp = trajs(sl==usl(i) & al==ual(j) & pl==upl(k),:);
            [aVec,mainPath,mainWeight] = findAtomVector(X_tmp);
            aFeat = [aFeat, aVec];
            asl = [asl, usl(i)];
            aal = [aal, ual(j)];
            apl = [apl, upl(k)];            
        end
        fprintf('action %d/%d processed.\n',j,length(ual));
    end    
end
%     save hFeat300_action01_06_person01_26_scene010304_20131118t hFeat hsl hal hpl;
load ../expData/aFeat_action01_06_person01_26_scene010304_20140210t;

X2_train = aFeat(:,ismember(apl,trainingSet));
X2_validate = aFeat(:,ismember(apl,validationSet));
X2_test = aFeat(:,ismember(apl,testingSet));
y2_train = aal(:,ismember(apl,trainingSet));
y2_validate = aal(:,ismember(apl,validationSet));
y2_test = aal(:,ismember(apl,testingSet));


Cind = -5:5;
G = 10.^Cind;
C = 10.^Cind;
% G = 1e-4;
% C = 10;
% G = 1e0;
% C = 1e-1;
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
disp(gi);
disp(ci)
accuracyMat(gi,ci) = mean(accuracy);

    end
end
% save('accuracy_action01_06_person01_26_scene01_01_20131118','accuracy','y2_test','predict_label');



% 55
% end