# Get all the frames in each view video.

# PyTorch includes
import cv2
import os

videos_src_path = "/home/zhenyao/multi_people/video"
videos_save_path = "/home/zhenyao/multi_people/frames"

videos = os.listdir(videos_src_path)
videos = filter(lambda x: x.endswith('MP4'), videos)

for each_video in videos:
    print (each_video)

    # get the name of each video, and make the directory to save frames
    each_video_name, _ = each_video.split('.')

    if not os.path.exists(videos_save_path + '/' + each_video_name):
        os.makedirs(videos_save_path + '/' + each_video_name)

        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

        # get the full path of each video, which will open the video
        each_video_full_path = os.path.join(videos_src_path, each_video)
        cap = cv2.VideoCapture(each_video_full_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print ("fps: ", fps)

        frame_count = 1
        success = True
        while(success):

            success, frame = cap.read()
            # want all the frames
            # if success == False:
            #     break

            # height, width, _ = frame.shape
            # frame = frame[220:height - 220, 448:width - 448]
            print ('Print a new frame: ', success)

            # # only want the first 1800 frames
            # if frame_count == 1800:
            #     success = False

            cv2.imwrite(each_video_save_full_path + "%d.jpeg" % frame_count, frame)
            frame_count += 1

        cap.release()
