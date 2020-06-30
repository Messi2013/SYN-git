import os
import cv2

mask_src_path = "/home/xinyi/Dataset/Syn/person_mask/cam2/"

frame_src_path = "/home/xinyi/Dataset/Syn/frames_clip_group/cam2/"
frame_save_path = "/home/xinyi/Dataset/Syn/frames_mask/cam2/"

flowx_src_path = "/home/xinyi/Dataset/Syn/flows_clip_group/cam2/"
flowx_save_path = "/home/xinyi/Dataset/Syn/flow_mask/cam2/"
flowy_src_path = "/home/xinyi/Dataset/Syn/flows_clip_group/cam2/"
flowy_save_path = "/home/xinyi/Dataset/Syn/flow_mask/cam2/"

mask_videos = os.listdir(mask_src_path)
mask_videos.sort(key=lambda x: int(x[2:]))
frame_videos = os.listdir(frame_src_path)
frame_videos.sort(key=lambda x: int(x[2:]))
flowx_videos = os.listdir(flowx_src_path)
flowx_videos.sort(key=lambda x: int(x[2:]))
flowy_videos = os.listdir(flowy_src_path)
flowy_videos.sort(key=lambda x: int(x[2:]))

for i in range(0, 7):
    mask_group = os.listdir(mask_src_path + mask_videos[i])
    mask_group.sort()
    frame_group = os.listdir(frame_src_path + frame_videos[i])
    frame_group.sort()
    flowx_group = os.listdir(flowx_src_path + flowx_videos[i] + '/flow_x/')
    flowx_group.sort()
    flowy_group = os.listdir(flowy_src_path + flowy_videos[i] + '/flow_y/')
    flowy_group.sort()

    for j in range(0, 14):
        mask_img = os.listdir(mask_src_path+mask_videos[i] + '/' + mask_group[j])
        mask_img.sort(key=lambda x: int(x[:-4]))
        frame_img = os.listdir(frame_src_path + frame_videos[i] + '/' + frame_group[j])
        frame_img.sort(key=lambda x: int(x[:-4]))
        flowx_img = os.listdir(flowx_src_path + flowx_videos[i] + '/flow_x/' + flowx_group[j])
        flowx_img.sort(key=lambda x: int(x[:-4]))
        flowy_img = os.listdir(flowy_src_path + flowy_videos[i] + '/flow_y/' + flowy_group[j])
        flowy_img.sort(key=lambda x: int(x[:-4]))

        for k in range(120):
            mask_path = mask_src_path + mask_videos[i] + '/' + mask_group[j] + '/' + mask_img[k]
            frame_path = frame_src_path + frame_videos[i] + '/' + frame_group[j] + '/' + frame_img[k]
            flowx_path = flowx_src_path + flowx_videos[i] + '/flow_x/' + flowx_group[j] + '/' + flowx_img[k]
            flowy_path = flowy_src_path + flowy_videos[i] + '/flow_y/' + flowy_group[j] + '/' + flowy_img[k]

            if not os.path.exists(frame_save_path + frame_videos[i] + '/' + frame_group[j]):
                os.makedirs(frame_save_path + frame_videos[i] + '/' + frame_group[j])
            if not os.path.exists(flowx_save_path + flowx_videos[i] + '/flow_x/' + flowx_group[j]):
                os.makedirs(flowx_save_path + flowx_videos[i] + '/flow_x/' + flowx_group[j])
            if not os.path.exists(flowy_save_path + flowy_videos[i] + '/flow_y/' + flowy_group[j]):
                os.makedirs(flowy_save_path + flowy_videos[i] + '/flow_y/' + flowy_group[j])

            save_path_frame = frame_save_path + frame_videos[i] + '/' + frame_group[j] + '/' + frame_img[k]
            save_path_flowx = flowx_save_path + flowx_videos[i] + '/flow_x/' + flowx_group[j] + '/' + flowx_img[k]
            save_path_flowy = flowy_save_path + flowy_videos[i] + '/flow_y/' + flowy_group[j] + '/' + flowy_img[k]

            mask = cv2.imread(mask_path)
            frame = cv2.imread(frame_path)
            flowx = cv2.imread(flowx_path)
            flowy = cv2.imread(flowy_path)

            frame_mask = cv2.bitwise_and(mask, frame)
            flowx_mask = cv2.bitwise_and(mask, flowx)
            flowy_mask = cv2.bitwise_and(mask, flowy)

            cv2.imwrite(save_path_frame, frame_mask)
            cv2.imwrite(save_path_flowx, flowx_mask)
            cv2.imwrite(save_path_flowy, flowy_mask)

