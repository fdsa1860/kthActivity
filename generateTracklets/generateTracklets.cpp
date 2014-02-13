#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <unistd.h>

int main()
{
std::string cmd;
std::string execute = "/home/xikang/research/code/kthActivity/3rdParty/dense_trajectory_release_v1.2/release/DenseTrack";
std::string outputDir = "/home/xikang/research/code/kthActivity/trackletsData";
std::string videoDir = "/home/xikang/research/data/KTH/activityData";
std::string action;
for(int k=0;k<6;++k)
{
	switch(k)
	{
		case 0:
			action = "boxing";
			break;
		case 1:
			action = "handclapping";
			break;
		case 2:
			action = "handwaving";
			break;
		case 3:
			action = "jogging";
			break;
		case 4:
			action = "running";
			break;
		case 5:
			action = "walking";
			break;
		default:
			std::cout<<"the action name is not found."<<std::endl;
	}
	for(int i=1;i<26;++i)
		for(int j=1;j<5;++j)
		{
			std::string video;
			std::string output;
			char tmp[50];
			sprintf(tmp,"person%02d_%s_d%d_uncomp.avi",i,action.c_str(),j);
			video = videoDir + '/' + action + '/' + tmp;
			sprintf(tmp,"person%02d_%s_d%d_uncomp_features.gz",i,action.c_str(),j);
			output = outputDir + '/' + tmp;
			cmd = execute + " " + video + " | gzip > " + output;
			system(cmd.c_str());
		}
}


}
