
function HH = getHH(features,opt)

% s = size(features{1});

% idx = randi(s(1),[500,1]);
% Hsize = 1*length(idx);
% Hsize = 540;
% Hsize = 1*s(1);
Hsize = 8;


if ~exist('opt','var')
    opt.H_structure = 'HHt';
    opt.metric = 'JLD';
end

HH = cell(1,size(features,2));
for i=1:size(features,2)
    t = reshape(features(:,i),2,[]);
%     t = diff(T, [], 2);
%     if size(t,1)>1000, t=t(idx,:); end % debug only, comment out in release code
    if strcmp(opt.H_structure,'HtH')
        nc = Hsize;
        nr = size(t,1)*(size(t,2)-nc+1);
        if nr<1, error('hankel size is too large.\n'); end
        Ht = hankel_mo(t,[nr nc]);
        HHt = Ht' * Ht;
    elseif strcmp(opt.H_structure,'HHt')
        nr = floor(Hsize/size(t,1))*size(t,1);
        nc = size(t,2)-floor(nr/size(t,1))+1;
        if nc<1, error('hankel size is too large.\n'); end
        Ht = hankel_mo(t,[nr nc]);
        HHt = Ht * Ht';
    end
    HHt = HHt / norm(HHt,'fro');
    if strcmp(opt.metric,'JLD') || strcmp(opt.metric,'JLD_denoise') ...
            || strcmp(opt.metric,'AIRM')
        I = opt.sigma*eye(size(HHt));
        HH{i} = HHt + I;
    elseif strcmp(opt.metric,'binlong')
        HH{i} = HHt;
    end
end


% data = reshape(data,s(1)*s(2),s(3));



% count = 1;
% while count <= size(data,1)
%     if all(data(count,:)==0), data(count,:) = []; continue; end
%     count = count + 1;
% end
% t = diff(data,[],2);
% if strcmp(opt.H_structure,'HtH')
%     nc = Hsize;
%     nr = size(t,1)*(size(t,2)-nc+1);
%     if nr<1, error('hankel size is too large.\n'); end
%     Ht = hankel_mo(t,[nr nc]);
%     HHt = Ht' * Ht;
% elseif strcmp(opt.H_structure,'HHt')
%     nr = floor(Hsize/size(t,1))*size(t,1);
%     nc = size(t,2)-floor(nr/size(t,1))+1;
%     if nc<1, error('hankel size is too large.\n'); end
%     Ht = hankel_mo(t,[nr nc]);
%     HHt = Ht * Ht';
% end
% HHt = HHt / norm(HHt,'fro');
% %     HHt = t * t';
% I = 0.9*eye(size(HHt));
% HH = HHt + I;

end