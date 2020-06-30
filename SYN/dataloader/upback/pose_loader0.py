# PyTorch includes
import ipdb
import os
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import resize
# Custom includes
from dataloader import *


def bboximg(img_path_cam_rgb,img_path_cam_bbox):
    img_cam = cv2.imread(img_path_cam_rgb)

    bbox_cam = open(img_path_cam_bbox)
    next(bbox_cam)
    for b1 in bbox_cam:
        b1 = b1.replace('\n', '').split(' ')
        w1, h1, w2, h2 = int(b1[0]), int(b1[1]), int(b1[2]), int(b1[3])

    if (h2 - h1) < 3 * (w2 - w1):
        img = img_cam[h1 - (3 * (w2 - w1) - (h2 - h1)) / 2:h1 + (3 * (w2 - w1) + (h2 - h1)) / 2, w1:w2, :]
    else:
        img = img_cam[h1:h2, w1 - ((h2 - h1) / 3 - (w2 - w1)) / 2:w1 + ((h2 - h1) / 3 + (w2 - w1)) / 2, :]

    img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_CUBIC)
    return img

def bboxflow(img_path_cam_rgb,img_path_cam_bbox):
    img_cam = cv2.imread(img_path_cam_rgb)

    bbox_cam = open(img_path_cam_bbox)
    next(bbox_cam)
    for b1 in bbox_cam:
        b1 = b1.replace('\n', '').split(' ')
        w1, h1, w2, h2 = int(b1[0]), int(b1[1]), int(b1[2]), int(b1[3])

    if (h2 - h1) < 3 * (w2 - w1):
        imgx = img_cam[h1 - (3 * (w2 - w1) - (h2 - h1)) / 2:h1 + (3 * (w2 - w1) + (h2 - h1)) / 2,
               w1:w2]
    else:
        img = img_cam[h1:h2, w1 - ((h2 - h1) / 3 - (w2 - w1)) / 2:w1 + ((h2 - h1) / 3 + (w2 - w1)) / 2]

    img = cv2.resize(img, (128, 384), interpolation=cv2.INTER_CUBIC)
    return img

class SYN(Dataset):
    """my own dataset to test"""

    def __init__(self, train=True,
                 db_root_dir=None,
                 transform=None,
                 epoch=None,
                 batch=None,
                 clip_lenth=8):

        self.train = train
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.epoch = epoch
        self.batch = batch
        self.clip_len = clip_lenth
        self.downsample = 1
        self.mode = 2

        # syn path
        # if self.train:
        #     self.video_cam1_path = os.path.join(db_root_dir, 'poses_group/cam1')
        #     self.video_cam2_path = os.path.join(db_root_dir, 'poses_group/cam2')
        #     name_list = ['0','1','2','4','5','7','8','9','11','12']
        # else:
        #     self.video_cam1_path = os.path.join(db_root_dir, 'poses_group/cam1_test')
        #     self.video_cam2_path = os.path.join(db_root_dir, 'poses_group/cam2_test')
        #     name_list = ['3','6','10','13']

        # zw path
        # if self.train:
        #     self.video_cam1_path = os.path.join(db_root_dir, 'train/cam1_pose')
        #     self.video_cam2_path = os.path.join(db_root_dir, 'train/cam2_pose')
        #     name_list = ['1','2','3','4','5','6','7','8','9','10']
        # else:
        #     self.video_cam1_path = os.path.join(db_root_dir, 'test/cam1_pose')
        #     self.video_cam2_path = os.path.join(db_root_dir, 'test/cam2_pose')
        #     name_list = ['1','2','3','4']

        # mtp path
        if self.train:#display false usually even if L56, maybe L56 changes L49 train's status of True
            self.video_cam1_path = os.path.join(db_root_dir, 'SYN_DB/MPV/train/pose1/cam1')
            self.video_cam2_path = os.path.join(db_root_dir, 'SYN_DB/MPV/train/pose1/cam2')
            name_list = ['0','5','7']#the order is only the frame clip sequence, such as 1-240, 241-480,..., the group are same, only different tome. 
            # name_list = ['10']
        else:
            self.video_cam1_path = os.path.join(db_root_dir, 'SYN_DB/MPV/train/pose1/cam1')
            self.video_cam2_path = os.path.join(db_root_dir, 'SYN_DB/MPV/train/pose1/cam2')
            name_list = ['3','5','7']

        # ipdb.set_trace()
        self.video_cam1 = np.sort(os.listdir(self.video_cam1_path))
        self.video_cam2 = np.sort(os.listdir(self.video_cam2_path))

        self.img_cam1_list = []
        self.img_cam2_list = []

        self.group_name_path_cam1 = []
        self.group_name_path_cam2 = []

        for n in range(len(self.video_cam1)): 
            
            each_video_path_cam1 = os.path.join(self.video_cam1_path, self.video_cam1[n])
            each_video_path_cam2 = os.path.join(self.video_cam2_path, self.video_cam2[n])

            #group_name = os.listdir(each_video_path_cam1)
            #group_name.sort(key=int)

            #self.group_name = group_name

            group_name_cam1 = os.listdir(each_video_path_cam1)
            group_name_cam2 = os.listdir(each_video_path_cam2)
         
            self.group_name_path_cam1.append(each_video_path_cam1)
            self.group_name_path_cam2.append(each_video_path_cam2)

            self.img_cam1_list.append(group_name_cam1)
            self.img_cam2_list.append(group_name_cam2)

            #for j in range(len(name_list)):
            #    group_name_cam1 = os.listdir(os.path.join(each_video_path_cam1, name_list[j]))
            #    group_name_cam1.sort(key=lambda x: int(x[:-4]))
            #    group_name_cam2 = os.listdir(os.path.join(each_video_path_cam2, name_list[j]))
            #    group_name_cam2.sort(key=lambda x: int(x[:-4]))

            #    self.group_name_path_cam1.append(os.path.join(each_video_path_cam1, name_list[j]))
            #    self.group_name_path_cam2.append(os.path.join(each_video_path_cam2, name_list[j]))

            #    self.img_cam1_list.append(group_name_cam1)
            #    self.img_cam2_list.append(group_name_cam2)
            
        # print('self.img_cam1_list',self.img_cam1_list)
        # print('self.img_cam2_list',self.img_cam2_list)
        # print('self.group_name_path_cam1',self.group_name_path_cam1)
        # print('self.group_name_path_cam2',self.group_name_path_cam2)

    def __getitem__(self, idx):

        if self.train:
            video_item = (idx // self.clip_len) * self.batch
        else:
            video_item = (idx // (self.clip_len*2)) * self.batch
        # ipdb.set_trace()
        for video_num in range(video_item, video_item + self.batch): #video_num is the sequence NO. under scratch-pose-npy's foulder

            img_num = len(os.listdir(self.group_name_path_cam1[video_num])) / self.downsample#sample with an interval the 240 npys in a foulder
            print("img_num",img_num)
            # offset = 0 # np.random(0, 10)
            offset = np.random.random_integers(-4, 3)#only 0-7 can run without bug, if 8 will exceed the range of offset (clip-len=8)
            # if offset == 8:
            #     offset = 255
            print("offset", offset)
            # random get 30 frame in the a group (total 210 group) for each cam
            if self.train:
                # TODO: change the bounds
                # ipdb.set_trace()
                first_possible_index = self.clip_len/2 # =4
                last_possible_index = img_num-self.clip_len*3/2 # =240-12=228
                t = random.randint(first_possible_index, last_possible_index)
            else:
                if idx % (self.clip_len*2) < self.clip_len:
                    t = int(self.clip_len/2)
                else:
                    t = int(self.clip_len)
            print("t",t)
            print("self.clip_len",self.clip_len)

            clip_cam1 = []
            clip_cam2 = []
            clip_cam1_rgb = []
            clip_cam2_rgb = []
            clip_cam1_flowx = []
            clip_cam2_flowx = []
            clip_cam1_flowy = []
            clip_cam2_flowy = []

            # ipdb.set_trace()
            for k in range(self.clip_len):
                # Maybe:Helge require to print
                # print(len(self.img_cam1_list[video_num][::self.downsample]))
                # print(self.downsample, t, self.clip_len, k)
                img_path_cam1 = os.path.join(self.group_name_path_cam1[video_num], 
                                             self.img_cam1_list[video_num][::self.downsample][t:t + self.clip_len][k])
                print('img_path_cam1 ', img_path_cam1)
                print("list", len(self.img_cam2_list[video_num][::self.downsample]))
                print("index",t + offset, t + self.clip_len + offset)
                
                img_path_cam2 = os.path.join(self.group_name_path_cam2[video_num],
                                             self.img_cam2_list[video_num][::self.downsample][t + offset:t + self.clip_len + offset][k])
                print('img_path_cam2 ', img_path_cam2)
                # debugging code
                entire_sequence1 = self.img_cam1_list[video_num][::self.downsample]
                entire_sequence2 = self.img_cam2_list[video_num][::self.downsample]
                # lets assume len(entire_sequence1)=200
                subsequencee_sequence1 = entire_sequence1[t:t + self.clip_len]
                subsequencee_sequence2 = entire_sequence2[t + offset:t + self.clip_len + offset]
                # problem if (t + self.clip_len + offset > 200), then len(subsequencee_sequence2)<len(subsequencee_sequence1)
                # Helge
                # print("subsequencee_sequence1", len(subsequencee_sequence1))
                # print("subsequencee_sequence2", len(subsequencee_sequence2))

                portion = os.path.splitext(img_path_cam1)
                # ipdb.set_trace()
                img_path_cam1 = portion[0] + ".npy"

                if self.mode % 2 == 0:
                    # print('img_path_cam1',img_path_cam1)
                    # print('img_path_cam2',img_path_cam2)
                    img1 = np.load(img_path_cam1, allow_pickle=True)
                    #img1.permute(2, 0, 1).size()
                    img1 =  img1.transpose(2, 1, 0)
                    # print('img1=', img1.shape)
                    img_cam1 = np.array(img1, dtype=np.float32)
                    clip_cam1.append(img_cam1)
                    img2 = np.load(img_path_cam2)
                    img2 =  img2.transpose(2, 1, 0)
                    img_cam2 = np.array(img2, dtype=np.float32)
                    clip_cam2.append(img_cam2)

                    cam1 = np.stack(clip_cam1)
                    cam2 = np.stack(clip_cam2)


                if self.mode % 3 ==0:
                    img_path_cam1_rgb = img_path_cam1[:33]+str('frames')+img_path_cam1[38:-3]+str('jpg')
                    img_path_cam2_rgb = img_path_cam2[:33]+str('frames')+img_path_cam2[38:-3]+str('jpg')
                    img_path_cam1_bbox = img_path_cam1[:33]+str('bbox')+img_path_cam1[38:-3]+str('txt')
                    img_path_cam2_bbox = img_path_cam1[:33]+str('bbox')+img_path_cam2[38:-3]+str('txt')
                    # ipdb.set_trace()
                    img1 =bboximg(img_path_cam1_rgb, img_path_cam1_bbox)
                    clip_cam1_rgb.append(img1)

                    img2 = bboximg(img_path_cam2_rgb, img_path_cam2_bbox)
                    clip_cam2_rgb.append(img2)

                    cam1_rgb = np.stack(clip_cam1_rgb)
                    cam2_rgb = np.stack(clip_cam2_rgb)

                if self.mode % 5 == 0:
                    img_path_cam1_flowx = img_path_cam1[:33]+str('flows')+img_path_cam1[38:59]+str('/flow_x')+img_path_cam1[-10:-3]+str('jpg')
                    img_path_cam1_flowy = img_path_cam1[:33]+str('flows')+img_path_cam1[38:59]+str('/flow_y')+img_path_cam1[-10:-3]+str('jpg')
                    img_path_cam2_flowx = img_path_cam2[:33]+str('flows')+img_path_cam2[38:59]+str('/flow_x')+img_path_cam2[-10:-3]+str('jpg')
                    img_path_cam2_flowy = img_path_cam2[:33]+str('flows')+img_path_cam2[38:59]+str('/flow_y')+img_path_cam2[-10:-3]+str('jpg')
                    img_path_cam1_bbox = img_path_cam1[:33]+str('bbox')+img_path_cam1[38:-3]+str('txt')
                    img_path_cam2_bbox = img_path_cam1[:33]+str('bbox')+img_path_cam2[38:-3]+str('txt')

                    img1x = bboxflow(img_path_cam1_flowx, img_path_cam1_bbox)
                    clip_cam1_flowx.append(img1x)
                    img1y = bboxflow(img_path_cam1_flowy, img_path_cam1_bbox)
                    clip_cam1_flowy.append(img1y)

                    img2x = bboxflow(img_path_cam2_flowx, img_path_cam2_bbox)
                    clip_cam2_flowx.append(img2x)
                    img2y = bboxflow(img_path_cam2_flowy, img_path_cam2_bbox)
                    clip_cam2_flowy.append(img2y)

                    cam1_flowx = np.stack(clip_cam1_flowx)
                    cam2_flowx = np.stack(clip_cam2_flowx)
                    cam1_flowy = np.stack(clip_cam1_flowy)
                    cam2_flowy = np.stack(clip_cam2_flowy)

                lbl = offset + self.clip_len/2

                print("lbl", lbl)

        if self.mode == 2:
            return cam1, cam2, lbl # currently: heatmap1, heatmap2, label
        if self.mode == 3:
            return cam1_rgb, cam2_rgb, lbl
        if self.mode == 5:
            return cam1_flowx, cam2_flowx, cam1_flowy, cam2_flowy, lbl # TODO2: optical flow on top
        if self.mode == 6:
            return cam1, cam2, cam1_rgb, cam2_rgb, lbl # TODO1: return this: heatmap1, heatmap2, image1, image2, label
        if self.mode == 10:
            return cam1, cam2, cam1_flowx, cam2_flowx, cam1_flowy, cam2_flowy, lbl
        if self.mode == 15:
            return cam1_rgb, cam2_rgb, cam1_flowx, cam2_flowx, cam1_flowy, cam2_flowy, lbl
        if self.mode == 30:
            return cam1, cam2, cam1_rgb, cam2_rgb, cam1_flowx, cam2_flowx, cam1_flowy, cam2_flowy, lbl

    def __len__(self):
        if self.train:
            return self.clip_len*len(self.img_cam1_list)*self.batch
        else:
            return self.clip_len*2*len(self.img_cam1_list)*self.batch


if __name__ == '__main__':

    # for epoch in range(1,300):
    transforms = transforms.Compose(
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    )

    dataset = SYN(db_root_dir='/scratch/meil3/SYN_DB',
                  train=True, transform=transforms, epoch=1, batch=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print('len(dataloader)', len(dataloader))
    for i, (cam1, cam2, lbl) in enumerate(dataloader):
        # loss = train(cam1,cam2,lbl)
        # print (cam2.shape)
        if i == 2:
            break
