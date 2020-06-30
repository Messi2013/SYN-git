import os
import numpy as np
import cv2
import shutil

cam = "cam2/v007/"
pic = "13/"
a = 390
b = 300
height = 576
weight = 576

videos_frames_src_path = "/home/xinyi/Dataset/Syn/frames_group/"+cam+pic
videos_frames_save_path = "/home/xinyi/Dataset/Syn/frames_clip_group/"+cam+pic
videos_flows_x_src_path = "/home/xinyi/Dataset/Syn/flows_group/"+cam+"flow_x/"+pic
videos_flows_x_save_path = "/home/xinyi/Dataset/Syn/flows_clip_group/"+cam+"flow_x/"+pic
videos_flows_y_src_path = "/home/xinyi/Dataset/Syn/flows_group/"+cam+"flow_y/"+pic
videos_flows_y_save_path = "/home/xinyi/Dataset/Syn/flows_clip_group/"+cam+"flow_y/"+pic

if not os.path.exists(videos_frames_save_path):
    os.makedirs(videos_frames_save_path)
if not os.path.exists(videos_flows_x_save_path):
    os.makedirs(videos_flows_x_save_path)
if not os.path.exists(videos_flows_y_save_path):
    os.makedirs(videos_flows_y_save_path)

video_frames = np.sort(os.listdir(videos_frames_src_path))
for i in range(len(video_frames)):
    frame = cv2.imread(videos_frames_src_path+video_frames[i])
    flow_x = cv2.imread(videos_flows_x_src_path+video_frames[i])
    flow_y = cv2.imread(videos_flows_y_src_path+video_frames[i])
    frame = frame[a:a+height, b:b+weight]
    flow_x = flow_x[a:a+height, b:b+weight]
    flow_y = flow_y[a:a+height, b:b+weight]
    cv2.imwrite(videos_frames_save_path+video_frames[i], frame)
    cv2.imwrite(videos_flows_x_save_path+video_frames[i], flow_x)
    cv2.imwrite(videos_flows_y_save_path+video_frames[i], flow_y)
# for i in range(len(video_frames)):
#
#     local_path = os.path.join(videos_frames_src_path, video_frames[i])
#
#     each_video_frame_full_path.append(os.listdir(local_path))
#     each_frame = (filter(lambda x: x.endswith('jpg'), each_video_frame_full_path[i]))
#     each_frame.sort(key=lambda x: int(x[:-4]))
#     each_frame_list.append(each_frame)
#
#     # make group, 100 frames in a group
#     for j in range(0, len(each_frame), 120):
#
#         # get the name of each video, and make the directory to save group frames
#         each_group_name = str(j/120)
#         if not os.path.exists(videos_frames_save_path + video_frames[i] + '/' + each_group_name):
#             os.makedirs(videos_frames_save_path + video_frames[i] + '/' + each_group_name)
#
#         each_group_save_full_path = os.path.join(videos_frames_save_path + video_frames[i] + '/', each_group_name) + '/'
#
#         group_list.append(each_frame[j:j + 120])
#         for k in range(len(group_list[j/120])):
#             shutil.copy(local_path+'/'+str(group_list[j/120][k]), each_group_save_full_path+str(group_list[j/120][k]))
#         print("Finish group", j/120)
#
#     print(i+1, " Done")
