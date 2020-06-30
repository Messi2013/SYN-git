# Package include
import os
import timeit
import numpy as np
import shutil

# PyTorch includes
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Custom includes
from my_path import Path
# from dataloader.Syn_loader import SYN
from dataloader.pose_loader import SYN
from network import *
from tensorboardX import SummaryWriter

# Device configuration
device_ids = [0]
# torch.backends.cudnn.benchmark = True

# Hyper parameters
num_epoch = 501
learning_rate = 1e-4
weight_decay = 5e-4
resume = '/home/zhenyao/SYN_CMP2/model/vis/best_checkpoint.pth'
evaluate = True
clip_len = 8

# Path
db_root_dir = Path.db_root_dir()

# Transform
transforms = transforms.Compose(
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
)


def main():
    best_prec = -1

    save_dir = Path.save_root_dir()
    save_dir = save_dir+"/vis"
    train_writer = SummaryWriter(os.path.join(save_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(save_dir, 'test'))

    # Test Dataloader
    test_dataset = SYN(db_root_dir=db_root_dir,
                       train=False, transform=transforms, epoch=None, batch=1, clip_lenth=clip_len)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Build model
    print ("==================>Build model.")
    model = SynNet(clip_len).cuda(device_ids[0])

    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(),learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)

    # data parallel
    # model = nn.DataParallel(model, device_ids=device_ids)
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print ("==================>Loading checkpoint found at '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            # best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print ("==================>Loaded checkpoint '{}' (epoch {})".format(resume, start_epoch))
        else:
            print ("==================>No checkpoint found at '{}'".format(resume))

    if evaluate:
        test(test_loader, model, test_writer)

# Test the model
def test(test_loader, model, test_writer):
    print ("==================>Start Testing")
    test_step = len(test_loader)
    correct_test = 0
    acc = 0
    for test_num, (cam1, cam2, lbl) in enumerate(test_loader):
        # test data
        cam1 = Variable(cam1).cuda(device_ids[0])
        cam2 = Variable(cam2).cuda(device_ids[0])
        lbl = Variable(lbl).cuda(device_ids[0])

        # compute the output
        with torch.no_grad():
            output_dict = model(cam1, cam2, clip_len)

        pred_test = get_pred(output_dict['classification'])


        acc += (pred_test - float(lbl))



        torch.cuda.empty_cache()

    # measure the accuracy
    acc =  acc / len(test_loader)


    print ("The test accuracy is {:.4f}".format(acc))




def get_pred(result):
    result = torch.squeeze(result.cpu())
    result = result.detach().numpy()
    pred = np.argmax(result)

    return pred


def offset_loss(input, target):

    input = F.softmax(input, 1)
    input = torch.squeeze(input.cpu()).detach().numpy()
    print (input)
    # input = input[0]

    out = np.zeros(input.shape)
    for i in range(len(input)):
        out[i] = input[i] * abs(float(target) - i)
    result = out.sum()
    result = torch.from_numpy(np.array([result])).type(torch.FloatTensor).cuda()
    result = Variable(result, requires_grad=True)

    return result

def P_loss(input, target):
    result = abs(input-target)/clip_len

    return result


if __name__ == '__main__':
    main()
