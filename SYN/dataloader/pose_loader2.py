# PyTorch includes
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

        # rgb path
        if self.train:
            self.video_cam1_path = os.path.join(db_root_dir, 'poses_group/cam1')
            self.video_cam2_path = os.path.join(db_root_dir, 'poses_group/cam2')
            name_list = ['0','1','2','4','5','7','8','9','11','12']
        else:
            self.video_cam1_path = os.path.join(db_root_dir, 'poses_group/cam1_test')
            self.video_cam2_path = os.path.join(db_root_dir, 'poses_group/cam2_test')
            name_list = ['3','6','10','13']

        self.video_cam1 = np.sort(os.listdir(self.video_cam1_path))
        self.video_cam2 = np.sort(os.listdir(self.video_cam2_path))

        self.img_cam1_list = []
        self.img_cam2_list = []

        self.group_name_path_cam1 = []
        self.group_name_path_cam2 = []

        for n in range(len(self.video_cam1)):
            each_video_path_cam1 = os.path.join(self.video_cam1_path, self.video_cam1[n])
            each_video_path_cam2 = os.path.join(self.video_cam2_path, self.video_cam2[n])

            group_name = os.listdir(each_video_path_cam1)
            group_name.sort(key=int)

            self.group_name = group_name

            for j in range(len(group_name)):
                group_name_cam1 = os.listdir(os.path.join(each_video_path_cam1, name_list[j]))
                group_name_cam1.sort(key=lambda x: int(x[:-4]))
                group_name_cam2 = os.listdir(os.path.join(each_video_path_cam2, name_list[j]))
                group_name_cam2.sort(key=lambda x: int(x[:-4]))

                self.group_name_path_cam1.append(os.path.join(each_video_path_cam1, name_list[j]))
                self.group_name_path_cam2.append(os.path.join(each_video_path_cam2, name_list[j]))

                self.img_cam1_list.append(group_name_cam1)
                self.img_cam2_list.append(group_name_cam2)

    def __getitem__(self, idx):

        if self.train:
            video_item = (idx // self.clip_len) * self.batch
        else:
            video_item = (idx // (self.clip_len*2)) * self.batch

        for video_num in range(video_item, video_item + self.batch):

            img_num = len(os.listdir(self.group_name_path_cam1[video_num])) / self.downsample

            offset = idx % self.clip_len - self.clip_len/2

            # random get 30 frame in the a group (total 210 group) for each cam
            if self.train:
                t = random.randint(self.clip_len/2, img_num-self.clip_len*3/2)
            else:
                if idx % (self.clip_len*2) < self.clip_len:
                    t = self.clip_len/2
                else:
                    t = self.clip_len/2

            clip_cam1 = []
            clip_cam2 = []
            clip_cam1_rgb = []
            clip_cam2_rgb = []
            clip_cam1_flowx = []
            clip_cam2_flowx = []
            clip_cam1_flowy = []
            clip_cam2_flowy = []


            for k in range(self.clip_len):
                img_path_cam1 = os.path.join(self.group_name_path_cam1[video_num],
                                             self.img_cam1_list[video_num][::self.downsample][t:t + self.clip_len][k])
                img_path_cam2 = os.path.join(self.group_name_path_cam2[video_num],
                                             self.img_cam2_list[video_num][::self.downsample][t + offset:t + self.clip_len + offset][k])

                if self.mode % 2 == 0:
                    img1 = np.load(img_path_cam1)
                    img_cam1 = np.array(img1, dtype=np.float32)
                    clip_cam1.append(img_cam1)
                    img2 = np.load(img_path_cam2)
                    img_cam2 = np.array(img2, dtype=np.float32)
                    clip_cam2.append(img_cam2)

                    cam1 = np.stack(clip_cam1)
                    cam2 = np.stack(clip_cam2)


                if self.mode % 3 ==0:
                    img_path_cam1_rgb = img_path_cam1[:33]+str('frames')+img_path_cam1[38:-3]+str('jpg')
                    img_path_cam2_rgb = img_path_cam2[:33]+str('frames')+img_path_cam2[38:-3]+str('jpg')
                    img_path_cam1_bbox = img_path_cam1[:33]+str('bbox')+img_path_cam1[38:-3]+str('txt')
                    img_path_cam2_bbox = img_path_cam1[:33]+str('bbox')+img_path_cam2[38:-3]+str('txt')

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

        if self.mode == 2:
            return cam1, cam2, lbl
        if self.mode == 3:
            return cam1_rgb, cam2_rgb, lbl
        if self.mode == 5:
            return cam1_flowx, cam2_flowx, cam1_flowy, cam2_flowy, lbl
        if self.mode == 6:
            return cam1, cam2, cam1_rgb, cam2_rgb, lbl
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

    dataset = SYN(db_root_dir='/home/xinyi/Dataset/Syn',
                  train=True, transform=transforms, epoch=1, batch=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print(len(dataloader))
    for i, (cam1, cam2, lbl) in enumerate(dataloader):
        # loss = train(cam1,cam2,lbl)
        print (cam2.shape)
        if i == 2:
            break