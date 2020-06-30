import os
import numpy as np
import shutil

videos_frames_src_path = "/home/xinyi/Dataset/Syn/video_frames_original/cam2/"
videos_frames_save_path = "/home/xinyi/Dataset/Syn/frames_group/cam2/"

# get each video frames path
each_video_frame_full_path = []
each_frame_list = []
group_list = []

video_frames = np.sort(os.listdir(videos_frames_src_path))
for i in range(len(video_frames)):

    local_path = os.path.join(videos_frames_src_path, video_frames[i])

    each_video_frame_full_path.append(os.listdir(local_path))
    each_frame = (filter(lambda x: x.endswith('jpg'), each_video_frame_full_path[i]))
    each_frame.sort(key=lambda x: int(x[:-4]))
    each_frame_list.append(each_frame)

    # make group, 100 frames in a group
    for j in range(0, len(each_frame), 120):

        # get the name of each video, and make the directory to save group frames
        each_group_name = str(j/120)
        if not os.path.exists(videos_frames_save_path + video_frames[i] + '/' + each_group_name):
            os.makedirs(videos_frames_save_path + video_frames[i] + '/' + each_group_name)

        each_group_save_full_path = os.path.join(videos_frames_save_path + video_frames[i] + '/', each_group_name) + '/'

        group_list.append(each_frame[j:j + 120])
        for k in range(len(group_list[j/120])):
            shutil.copy(local_path+'/'+str(group_list[j/120][k]), each_group_save_full_path+str(group_list[j/120][k]))
        print("Finish group", j/120)

    print(i+1, " Done")
