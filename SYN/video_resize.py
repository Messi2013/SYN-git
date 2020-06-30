import os

videos_src_path = "/home/xinyi/Dataset/Syn/video/cam1"
videos_save_path = "/home/xinyi/Dataset/Syn/video_resize/cam1"

videos = os.listdir(videos_src_path)
videos = filter(lambda x: x.endswith('mp4'), videos)

for video in videos:
    os.system('ffmpeg -i ' + str(os.path.join(videos_src_path, video)) +
              ' -vf scale=iw*0.5:ih*0.5 ' + str(os.path.join(videos_save_path, video)))
    print(str(video))
