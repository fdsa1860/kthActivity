
% read trajectories

function [trajs,al,pl,sl] = readTrajectory

trajPath = '/home/xikang/research/data/KTH/trackletsData';

trajs = [];
al = [];
pl = [];
sl = [];
for k=1:6
    switch k
        case 1,
            action = 'boxing';      
        case 2,
            action = 'handclapping';
        case 3,
            action = 'handwaving';
        case 4,
            action = 'jogging';
        case 5,
            action = 'running';
        case 6,
            action = 'walking';
        otherwise,
            error('the action name is not found.');
    end
    
    for i=1:25
        for j=1:4    
            fileName = sprintf('person%02d_%s_d%d_uncomp_features',i,action,j);
            traj = load(fullfile(trajPath,fileName));
            trajs = [trajs; traj];
            al = [al; k*ones(size(traj,1),1)];
            pl = [pl; i*ones(size(traj,1),1)];
            sl = [sl; j*ones(size(traj,1),1)];
            
        end
    end
% 55
end

end

