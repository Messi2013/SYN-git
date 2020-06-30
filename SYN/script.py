# 1. rename for the flow
# 2. get the first 1800 of each warp_flow

# PyTorch includes
import cv2
import os
import numpy as np

# 1. rename for the flow
img_src_path = "/home/zhenyao/multi_people/frames/frame2_new/"
img_src_path2 = "/home/zhenyao/multi_people/frames/frame2/v000/"

images = os.listdir(img_src_path)
images = filter(lambda x: x.endswith('jpg'), images)
# print len(images)

for img in images:
    if int(str(img)[:-4])%64 ==0:
        os.rename(str(img_src_path) + str(img), str(img_src_path2) + str(img))
    # print (str(img_src_path) + str(img))
    # print (str(img_src_path2) +str(int(str(img)[:-4])-299) + str(img)[-4:])

    print ('done ' + str(img))

# # 2. get the first 1800 of each warp_flow
# img_src_path = "/home/xinyi/Dataset/Syn/warp_flow/cam2/v014/flow_y/"
# images = os.listdir(img_src_path)
# images.sort(key=lambda x: int(x[7:-4]))
# print len(images)
#
# for img in images:
#     if int(img[7:-4]) > 1800:
#         os.remove(img_src_path + img)
#         print("done " + img)
#
# print len(images)

# # # 3. crop
# for i in range(0,14):
#     img_src_path = "/home/xinyi/Dataset/Syn/frames/cam2/v001/" + str(i) + '/'
#     bbox_src_path = "/home/xinyi/Dataset/Syn/frames/cam2_lbl/v001/" + str(i) + '/'
#     img_save_path = "/home/xinyi/Dataset/Syn/bbox/cam2/v001/"
#     img_group = os.listdir(img_src_path)
#     img_group.sort(key=lambda x: int(x[:-4]))
#     bbox_group = os.listdir(bbox_src_path)
#     bbox_group.sort(key=lambda x: int(x[:-4]))
#
#     for j in range(len(img_group)):
#         img_path = os.path.join(img_src_path, img_group[j])
#         bbox_path = os.path.join(bbox_src_path, bbox_group[j])
#         save_path = os.path.join(img_save_path + str(i) + '/', img_group[j])
#
#         if not os.path.exists(img_save_path + str(i) + '/'):
#             os.makedirs(img_save_path + str(i) + '/')
#
#         bbox = open(bbox_path)
#         next(bbox)
#         for b in bbox:
#             b = b.replace('\n', '').split(' ')
#             w1, h1, w2, h2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
#
#             img = cv2.imread(img_path)
#             if (h2-h1) < 3*(w2-w1):
#                 img = img[h1-(3*(w2-w1)-(h2-h1))/2:h1+(3*(w2-w1)+(h2-h1))/2, w1:w2]
#             else:
#                 img = img[h1:h2, w1-((h2-h1)/3-(w2-w1))/2:w1+((h2-h1)/3+(w2-w1))/2]
#             img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_CUBIC)
#
#             cv2.imwrite(save_path, img)
#         print ("done", j)


# # 4. list file name
# img_src_path = "/home/xinyi/Dataset/Syn/frames_group/cam1"
# videos = os.listdir(img_src_path)
# videos.sort(key=lambda x: int(x[2:]))
#
# for video in videos:
#     videos_path = os.path.join(img_src_path, video)
#     if os.path.isdir(videos_path):
#         groups = os.listdir(videos_path)
#         groups.sort(key=int)
#         for group in groups:
#             groups_path = os.path.join(videos_path, group)
#             img = os.listdir(groups_path)
#             img.sort(key=lambda x: int(x[:-4]))
#         print img

# # 5. change file names rgb
# img_src_path = "/home/xinyi/Dataset/Syn/video_frames_original/cam2"
# img_save_path = "/home/xinyi/Dataset/Syn/video_frames_original_1/cam2"
# videos = os.listdir(img_src_path)
# videos.sort(key=lambda x: int(x[2:]))
#
# for video in videos:
#     videos_path = os.path.join(img_src_path, video)
#
#     if not os.path.exists(img_save_path + '/' + video):
#         os.makedirs(img_save_path + '/' + video)
#
#     imgs = os.listdir(videos_path)
#     imgs.sort(key=lambda x: int(x[:-4]))
#     print ("len", len(imgs))
#     cnt = 0
#     for img in imgs:
#         oldname = videos_path + '/' + imgs[cnt]
#         newname = img_save_path + '/' + video + '/' + str(cnt+1) + '.jpg'
#         os.rename(oldname, newname)
#         # print (oldname, "--------->", newname)
#         cnt += 1

# # 6. change file names flow
# img_src_path = "/home/xinyi/Dataset/Syn/warp_flow/cam2"
# img_save_path = "/home/xinyi/Dataset/Syn/warp_flow_original/cam2"
# videos = os.listdir(img_src_path)
# videos.sort(key=lambda x: int(x[2:]))
# for video in videos:
#     videos_path = os.path.join(img_src_path, video)
#
#     flow_x_path = os.path.join(videos_path, 'flow_x')
#     flow_x = os.listdir(flow_x_path)
#     flow_x.sort(key=lambda x: int(x[7:-4]))
#
#     flow_y_path = os.path.join(videos_path, 'flow_y')
#     flow_y = os.listdir(flow_y_path)
#     flow_y.sort(key=lambda x: int(x[7:-4]))
#
#     if not os.path.exists(img_save_path + '/' + video + '/flow_x'):
#         os.makedirs(img_save_path + '/' + video + '/flow_x')
#     if not os.path.exists(img_save_path + '/' + video + '/flow_y'):
#         os.makedirs(img_save_path + '/' + video + '/flow_y')
#
#     print ("len_x", len(flow_x))
#     print ("len_y", len(flow_y))
#     cnt = 0
#     for img in range(0, 1680):
#         oldname_x = flow_x_path + '/' + flow_x[cnt]
#         newname_x = img_save_path + '/' + video + '/flow_x/' + str(cnt+1) + '.jpg'
#
#         oldname_y = flow_y_path + '/' + flow_y[cnt]
#         newname_y = img_save_path + '/' + video + '/flow_y/' + str(cnt+1) + '.jpg'
#         os.rename(oldname_x, newname_x)
#         os.rename(oldname_y, newname_y)
# #         print (oldname_x, "--------->", newname_x)
# #         print (oldname_y, "--------->", newname_y)
#         cnt += 1

# # 7. delete some images
# video_src_path = "/home/xinyi/Dataset/Syn/temp/"
# video_des_path = "/home/xinyi/Dataset/Syn/flow/cam1/"
#
# video_src = os.listdir(video_src_path)
# video_src.sort()
# video_des = os.listdir(video_des_path)
# video_des.sort()
#
# for video in video_src:
#     flow_x = video_des_path+video+"/flow_x/"
#     flow_y = video_des_path+video+"/flow_y/"
#     group_src_path = os.path.join(video_src_path, video)
#     group_src = os.listdir(group_src_path)
#     group_src.sort(key=int)
#     for group in group_src:
#         img_src_path = os.path.join(group_src_path, group)
#         img_src = os.listdir(img_src_path)
#         img_src.sort(key=lambda x: int(x[:-4]))
#
#         img_des_path = os.path.join(video_des_path, video) + '/' + str(group)
#         img_des = os.listdir(img_des_path)
#         img_des.sort(key=lambda x: int(x[:-4]))
#         img_diff = list(set(img_des).difference(set(img_src)))
#
        # cnt = 0
        # for i in range(len(img_des)):
        #     oldname = img_src_path + '/' + img_src[cnt]
        #     if not os.path.exists('/home/xinyi/Dataset/Syn/temp/' + video + '/' + str(group)):
        #         os.makedirs('/home/xinyi/Dataset/Syn/temp/' + video + '/' + str(group))
        #     newname = '/home/xinyi/Dataset/Syn/temp/' + video + '/' + str(group) + '/' + str(cnt+1) + '.jpg'
        #     cnt += 1
        #     os.rename(oldname, newname)
#
        # for img in img_diff:
        #     delete_path = os.path.join(img_des_path, img)
        #     os.remove(delete_path)

