% display trajectories

% set parameters
actionStr = 'boxing';
personNum = 1;
sceneNum = 1;

% load video
dataPath = '/home/xikang/research/data/KTH/activityData';
videoName = fullfile(dataPath,actionStr,sprintf('person%02d_%s_d%d_uncomp.avi',personNum,actionStr, sceneNum));
vid = mmread(videoName);

trajPath = '/home/xikang/research/data/KTH/trackletsData';
file = fullfile(trajPath,sprintf('person%02d_%s_d%d_uncomp_features2',personNum,actionStr, sceneNum));
traj = load(file);

% color space
c = 'bgrymckbgrymckbgrymck';

% determine the number of frames
nFrames = vid.nrFramesTotal;
% display video
frStart = 1;
frEnd = nFrames;
for i = frStart:frEnd
    % display frame i
    frame = vid.frames(i).cdata;
    imshow(frame); drawnow;
    % get trajectories of current frame
    currTraj = traj(traj(:,1)==i,2:end);
    if isempty(currTraj),  continue; end
    % draw trajectories
    hold on;
    X = currTraj(:,1:2:end);
    Y = currTraj(:,2:2:end);
    plot(X',Y','g');
    drawnow;
    hold off;
    i
end