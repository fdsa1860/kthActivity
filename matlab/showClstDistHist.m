% show cluster distance histogram

clear;
%% load data
file = 'seq_action01_06_person01_26_scene01_04_20131118f.mat';
load(fullfile('../expData',file));
V = trajs(:,2:end)';
al = reshape(al,1,[]);
pl = reshape(pl,1,[]);
sl = reshape(sl,1,[]);

% average filtering and get velocity
% h = [1; 1; 1]/3;
% Tx = T(1:2:end,:); Tx = conv2(Tx, h, 'valid'); Vx = diff(Tx);
% Ty = T(2:2:end,:); Ty = conv2(Ty, h, 'valid'); Vy = diff(Ty);
% V = zeros(size(Vx,1)*2,size(Vx,2)); V(1:2:end,:) = Vx; V(2:2:end,:) = Vy;
Vx = V(1:2:end,:);
Vy = V(2:2:end,:);
Vnorm = sum(sqrt(Vx.^2+Vy.^2));
X = bsxfun(@rdivide,V,Vnorm);

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

%% load centers
% load kmeansWords300_action01_06_person01_26_scene01_04_histDist_20150402f;
load ../expData/kmeansBinlong_w300_action01_06_person01_26_scene01_04_20150403f;
% load kmeansJLD_w300_action01_06_person01_26_scene01_04_20150403f;

label = sortLabel(params.label);
ind = params.rndInd;

centers = trainCenter{1};
trainData = X_train(:,ind);


%% compare
n = size(trainData,2);
d = size(trainData,1);
k = length(centers);

opt.metric = 'binlong';
% opt.metric = 'JLD';
nc = 8;
nr = (d/2-nc+1)*2;
HH_train = cell(1,n);
for i = 1:n
    H1 = hankel_mo(reshape(trainData(:,i),2,[]),[nr, nc]);
    H1_p = H1 / (norm(H1*H1','fro')^0.5);
    HH1 = H1_p * H1_p';
%     H1_p = H1 / norm(H1*H1','fro');
%     HH1 = H1_p * H1_p';
%     HH1 = HH1 / norm(HH1, 'fro');
    if strcmp(opt.metric,'JLD')
        HH_train{i} = HH1 + 1e-6 * eye(nr);
%         HH{i} = HH1;
    elseif strcmp(opt.metric,'binlong')
        HH_train{i} = HH1;
    end
end

if strcmp(opt.metric,'JLD')
    HH_centers = trainCenter{1};
elseif strcmp(opt.metric,'binlong')
    HH_centers = cell(1, k);
    for i = 1:k
        H1 = hankel_mo(reshape(centers(:,i),2,[]),[nr, nc]);
        H1_p = H1 / (norm(H1*H1','fro')^0.5);
        HH1 = H1_p * H1_p';
        if strcmp(opt.metric,'JLD')
            HH_centers{i} = HH1 + 1e-6 * eye(nr);
            %         HH{i} = HH1;
        elseif strcmp(opt.metric,'binlong')
            HH_centers{i} = HH1;
        end
    end
end

%% Compute distances
% tic
% D = zeros(n);
% for i = 1:n
%     for j = 1:n
%         if strcmp(opt.metric,'JLD')
%             HH1 = HH_train{i};
%             HH2 = HH_train{j};
%             D(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) -0.5*log(det(HH2));
%         elseif strcmp(opt.metric,'binlong')
%             D(j,i) = 2 - norm(HH_train{i}+HH_train{j},'fro');
%         end
%     end
% end
% toc

% tic
% D2 = zeros(k,n);
% for i = 1:n
%     for j = 1:k
%         if strcmp(opt.metric,'JLD')
%             HH1 = HH_train{i};
%             HH2 = HH_centers{j};
%             D2(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) -0.5*log(det(HH2));
%         elseif strcmp(opt.metric,'binlong')
%             D2(j,i) = 2 - norm(HH_train{i}+HH_centers{j},'fro');
%         end
%     end
% end
% toc

% tic
% D3 = zeros(k,k);
% for i = 1:k
%     for j = i:k
%         if strcmp(opt.metric,'JLD')
%             HH1 = HH_centers{i};
%             HH2 = HH_centers{j};
%             D3(j,i) = log(det((HH1+HH2)/2)) - 0.5*log(det(HH1)) -0.5*log(det(HH2));
%         elseif strcmp(opt.metric,'binlong')
%             D3(j,i) = 2 - norm(HH_centers{i}+HH_centers{j},'fro');
%         end
%     end
% end
% D3 = D3 + D3';
% toc

label1 = label;
HH_centers1 = HH_centers;
load ../expData/kmeansJLD_w300_action01_06_person01_26_scene01_04_20150403f;
HH_centers2 = trainCenter{1};
% for i = 1:length(HH_centers2)
%     HH_centers2{i} = HH_centers2{i} / norm(HH_centers2{i},'fro');
% end
label2 = sortLabel(params.label);
tic;D4 = HHdist(HH_centers1, HH_centers2, 'binlong');toc
tic;D5 = HHdist(HH_centers1, HH_centers2, 'JLD');toc
tic
D6 = zeros(k);
for i = 1:k
    for j = 1:k
        D6(i,j) = nnz(label1==i & label2==j) / nnz(label1==i | label2==j);
    end
end
toc

%% plot
c1 = 1;
c2 = 5;
ind1 = find(label==c1);
ind2 = find(label==c2);
scale = 0:1:100;
M = D;
% M = D/0.1;
A1 = M(ind1,ind1);
A2 = M(ind2,ind2);
A12 = M(ind1,ind2);
h1 = hist(A1(:),scale);
h2 = hist(A2(:),scale);
h12 = hist(A12(:),scale);
figure(1);
plot(scale,h1,'b');
hold on;
plot(scale,h2,'g');
plot(scale,h12,'r');
hold off;

M = D2;
B1 = M(c1,ind1);
B2 = M(c2,ind2);
B12 = M(c1,ind2);
B21 = M(c2,ind1);
g1 = hist(B1(:),scale);
g2 = hist(B2(:),scale);
g12 = hist(B12(:),scale);
g21 = hist(B21(:),scale);
figure(2);
plot(scale,g1,'b');
hold on;
plot(scale,g2,'g');
plot(scale,g12,'r');
plot(scale,g21,'m');
hold off;
xlabel('distance');
ylabel('histogram');
legend('all_1 to center_1','all_2 to center_2','all_1 to center_2','all_2 to center_1');

%% plot all inClusters
scale = 0:0.01:1;
M = D2;
h = zeros(k,length(scale));
for i = 1:k
    ind = find(label~=i);
    A = M(i,ind);
    h(i,:) = hist(A(:),scale);
end
figure;
hold on;
plot(scale,h);
hold off;
xlabel('distance');
ylabel('histogram');

%% distances between means
imagesc(log(D6));colorbar;
set(gca,'xtick',[1:30:300 300],'xTickLabel',{'1 (231)','31 (95)','61 (76)','91 (58)','121 (3)','151 (2)','181 (1)','211 (1)','241 (1)','271 (1)','300 (1)'});
set(gca,'ytick',[1:30:300 300],'yTickLabel',{'1 (421)','31 (88)','61 (45)','91 (32)','121 (22)','151 (13)','181 (10)','211 (7)','241 (5)','271 (3)','300 (1)'});
xlabel('JBLD cluster means: index (cluster size)');
ylabel('BM cluster means: index (cluster size)');

