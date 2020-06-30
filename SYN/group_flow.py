import os
import numpy as np
import cv2
import shutil

videos_frames_src_path = "/home/xinyi/Dataset/Syn/warp_flow_original/cam2/"
videos_frames_save_path = "/home/xinyi/Dataset/Syn/flows_group/cam2/"

# get each video frames path
each_video_frame_full_path = []
group_list = []

video_frames = np.sort(os.listdir(videos_frames_src_path))
for i in range(len(video_frames)):

    for group in range(0, 14):
        local_path_x = os.path.join(videos_frames_src_path, video_frames[i]) + "/flow_x/"
        each_frame_x = os.listdir(local_path_x)
        each_frame_x.sort(key=lambda x: int(x[:-4]))

        local_path_y = os.path.join(videos_frames_src_path, video_frames[i]) + "/flow_y/"
        each_frame_y = os.listdir(local_path_y)
        each_frame_y.sort(key=lambda x: int(x[:-4]))

        if not os.path.exists(videos_frames_save_path + video_frames[i] + '/flow_x/' + str(group)):
            os.makedirs(videos_frames_save_path + video_frames[i] + '/flow_x/' + str(group))
        if not os.path.exists(videos_frames_save_path + video_frames[i] + '/flow_y/' + str(group)):
            os.makedirs(videos_frames_save_path + video_frames[i] + '/flow_y/' + str(group))

        flow_x_path = os.path.join(videos_frames_save_path + video_frames[i] + '/flow_x/', str(group)) + '/'
        flow_y_path = os.path.join(videos_frames_save_path + video_frames[i] + '/flow_y/', str(group)) + '/'

        for k in range(len(each_frame_x)):
            # for crop
            # frame = cv2.imread(local_path + '/' + str(each_frame[k]))
            # height, width, _ = frame.shape
            # frame = frame[220:height-220, 448:width-448]
            src_x = local_path_x + '/' + str(each_frame_x[k])
            src_y = local_path_y + '/' + str(each_frame_y[k])
            save_x_path = flow_x_path + str(each_frame_x[k])
            save_y_path = flow_y_path + str(each_frame_y[k])
            shutil.copy(src_x, save_x_path)
            shutil.copy(src_y, save_y_path)
            # cv2.imwrite(save_path, frame)

        print("Finish group", group)

    print(i+1, " Done")
