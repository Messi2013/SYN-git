# Package include
import ipdb
import os
import timeit
import numpy as np
import shutil
# PyTorch includes
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

# Custom includes
from my_path import Path
# from dataloader.Syn_loader import SYN
from dataloader.pose_loader import SYN
from network import *
from tensorboardX import SummaryWriter

# Device configuration
device_ids = [0,1,2,3]
# torch.backends.cudnn.benchmark = True

# Hyper parameters
num_epoch = 501
learning_rate = 1e-4
weight_decay = 5e-4
resume = '/ubc'
# resume = '/ubc/cs/home/m/meil3/Code/SYN/model/vis/checkpoint-50.pth'
# resume = '/scratch/meil3/Result/SYN_model/nd/best_checkpoint.pth'
evaluate = False
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
    # ipdb.set_trace()
    save_dir = Path.save_root_dir()
    save_dir = save_dir+"/vis"
    train_writer = SummaryWriter(os.path.join(save_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(save_dir, 'test'))

    # Test Dataloader #why test module is prior to train module?
    test_dataset = SYN(db_root_dir=db_root_dir,
                       train=False, transform=transforms, epoch=None, batch=1, clip_lenth=clip_len)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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

    # Train the model
    print ("==================>Start Training")
    model.train() #pan
    for epoch in range(num_epoch):

        # Train Dataset
        train_dataset = SYN(db_root_dir=db_root_dir,
                            train=True, transform=transforms, epoch=epoch, batch=1, clip_lenth=clip_len)


        # Data loader
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        # one training epoch
        train_step = len(train_loader)
        loss_train_step = 0
        correct_train_d0 = 0
        correct_train_d1 = 0
        correct_train_d2 = 0
        correct_train_d3 = 0
        train_offset_epoch = 0

        # show the image visulization by 2.Writing to TensorBoard
        dataiter = iter(train_loader)
        # ipdb.set_trace()
        imagesCam1, imagesCam2, labels = dataiter.next()
        print('images_cam1_shape', imagesCam1.shape)#(1,8,19,82,46): cam_id x sequence_length x num channels x width x height
        print('images_cam2_shape', imagesCam2.shape)
        print('labels', labels)#pose loader.py: lbl = offset + self.clip_len/2 #lbl is the 'labels' here

        # cam1
        img_grid_hm_cam1_list = []
        for i in range(clip_len):
            # tensor dimension images: cam_id x sequence_length x num channels x width x height
            images_color_cam1 = imagesCam1[0,i,:,:,:] # TODO: make the heatmap a color image
            img_grid_hm_cam1 = heatmap2image(images_color_cam1)
            img_grid_hm_cam1_list.append(img_grid_hm_cam1)
        img_grid_hm_cam1_stack = torch.stack(img_grid_hm_cam1_list, dim=0)
        # img_grid_cam1 = torchvision.utils.make_grid(img_grid_hm_cam1_stack, pad_value=1, nrow=8) 
        # torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_hm_cam1_stack, pad_value=1, nrow=clip_len), '/ubc/cs/home/h/heyizhuo/epoch{}cam1.png'.format(epoch))
        # matplotlib_imshow(img_grid_cam1, one_channel=True)
        # train_writer.add_image('heatmap_images_cam1', img_grid_cam1)

        # cam2 
        img_grid_hm_cam2_list = []
        for i in range(clip_len):
            images_color_cam2 = imagesCam2[0,i,:,:,:] 
            img_grid_hm_cam2 = heatmap2image(images_color_cam2)
            img_grid_hm_cam2_list.append(img_grid_hm_cam2)
        img_grid_hm_cam2_stack = torch.stack(img_grid_hm_cam2_list, dim=0)
        # torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_hm_cam2_stack, pad_value=1, nrow=clip_len), '/ubc/cs/home/h/heyizhuo/epoch{}cam2.png'.format(epoch))

        img_grid_hm_two_cams_stack = torch.cat([img_grid_hm_cam1_stack, img_grid_hm_cam2_stack], dim=0)
        torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_hm_two_cams_stack, pad_value=1, nrow=clip_len), '/ubc/cs/home/m/meil3/Code/SYN/resultImg/cam1&2_epoch{}.png'.format(epoch))

        # measure time of each epoch
        start_time = timeit.default_timer()
        for i, (cam1, cam2, lbl) in enumerate(train_loader): #i is the inner loop of step[i/304],choose 304 random images to train
            # ipdb.set_trace()
            cam1 = Variable(cam1).cuda(device_ids[0])
            cam2 = Variable(cam2).cuda(device_ids[0])
            lbl = Variable(lbl).cuda(device_ids[0])

            # forward pass
            # ipdb.set_trace()
            output_dict = model(cam1, cam2, clip_len)#realted to L59 model-SYN
            print("~~~~~~~~the output_dict for these 2 imgs: ", output_dict['classification'])
            pred_train = get_pred(output_dict['classification'])
            loss = criterion(output_dict['classification'], lbl.long()) + P_loss(pred_train, float(lbl.long()))
            train_offset_rate = abs(pred_train - float(lbl))
            # loss = offset_loss(output_dict['classification'], lbl)
            train_offset_epoch += train_offset_rate

            if pred_train == int(lbl):
                correct_train_d0 += 1
            if abs(pred_train-int(lbl))<=1:
                correct_train_d1 += 1
            if abs(pred_train-int(lbl))<=2:
                correct_train_d2 += 1
            else:
                correct_train_d3 += 1
            # fuse losstrain_step
            loss_train_step += loss.item()
            # each_loss = loss.item() + train_offset_rate
            # loss_train_step += each_loss
            # loss = torch.from_numpy(np.array(loss_train_step, dtype=np.float64)).cuda()
            # loss = Variable(loss, requires_grad=True)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Label: {:d}, Result: {}'
                   .format(epoch + 1, num_epoch, i + 1, train_step, loss.item(), int(lbl), pred_train))

        # loss_train_step /= (8*50)
        loss_train_step /= (8 * 70)
        stop_time = timeit.default_timer()

        # measure the accuracy
        # train_hit = float(correct_train) / (8*50)
        # train_accuary = float(train_offset_epoch) / (8*50)
        train_hit0 = float(correct_train_d0) / (8*70)
        train_hit1 = float(correct_train_d1) / (8*70)
        train_hit2 = float(correct_train_d2) / (8*70)
        train_hit3 = float(correct_train_d3) / (8*70)
        train_distance = float(train_offset_epoch) / (8*70)

        # show the loss
        train_writer.add_scalar('loss', loss_train_step, epoch+1)
        train_writer.add_scalar('train_hit0', train_hit0, epoch+1)
        train_writer.add_scalar('train_hit1', train_hit1, epoch+1)
        train_writer.add_scalar('train_hit2', train_hit2, epoch+1)
        train_writer.add_scalar('train_hit3', train_hit3, epoch+1)
        train_writer.add_scalar('train_distance', train_distance, epoch+1)

        print ('Before new step loss: %.4f' % loss_train_step)
        print ('For this epoch, we use %.4f to finish' % (stop_time-start_time))
        print ("The train 0-dist-accuracy is %.4f" % train_hit0)
        print ("The train 1-dist-accuracy is %.4f" % train_hit1)
        print ("The train 2-dist-accuracy is %.4f" % train_hit2)
        print ("The average train distance is %.4f" % train_distance)

        # save model and the best checkpoint

        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        best_checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth')

        torch.save({
            'epoch': epoch + 1,
            'arch': 'SynNet',
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict()
        }, checkpoint_path)

        # save loss
        if epoch % 10 == 0:
            # evaluate on validation set
            prec, test_offset = test(test_loader, model, test_writer)

            # remember the best prec and save checkpoint
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
            if is_best:
                shutil.copy(checkpoint_path, best_checkpoint_path)

            f = open("train_loss.txt", "a+")
            f.write('epoch %d total training loss = %.4f\n' % (epoch, loss_train_step))
            f = open("test_prec.txt", "a+")
            f.write('epoch %d test prec = %.4f\n' % (epoch, prec))
            f = open("test_offset.txt", "a+")
            f.write('epoch %d test offset = %.4f\n' % (epoch, test_offset))


# Test the model
def test(test_loader, model, test_writer):
    model.eval() #pan
    print ("==================>Start Testing")
    test_step = len(test_loader)
    correct_test = 0
    test_offset_rate = 0
    for test_num, (cam1, cam2, lbl) in enumerate(test_loader):
        # test datan
        cam1 = Variable(cam1).cuda(device_ids[0])
        cam2 = Variable(cam2).cuda(device_ids[0])
        lbl = Variable(lbl).cuda(device_ids[0])

        # compute the output
        with torch.no_grad():
            output_dict = model(cam1, cam2, clip_len)

        pred_test = get_pred(output_dict['classification'])

        test_offset_rate += abs(pred_test - float(lbl))

        if pred_test == int(lbl):
            correct_test += 1
        # ipdb.set_trace()
        print ('Test Step [{}/{}], Label: {:d}, Result: {}'
               .format(test_num + 1, test_step, int(lbl), pred_test))

        torch.cuda.empty_cache()

        # measure the accuracy
        test_hit = float(correct_test) / len(test_loader)
        test_distance = float(test_offset_rate) / len(test_loader)

        test_writer.add_scalar('test_hit', test_hit, test_num + 1)
        test_writer.add_scalar('test_distance', test_distance, test_num + 1)
        print ("The test hit is {:.4f}".format(test_hit))
        print ("The test distance is {:.4f}".format(test_distance))

    return test_hit, test_distance


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

# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def heatmap2image(heatmap):
    C,H,W = heatmap.shape
    cmap = plt.get_cmap('hsv')
    img = torch.zeros(3,H,W).to(heatmap.device)
    for i in range(C):
        color = torch.FloatTensor(cmap(i * cmap.N // C)[:3]).reshape([-1,1,1]).to(heatmap.device)
        img = torch.max(img, color * heatmap[i]) # max in case of overlapping position of joints
    # heatmap and probability maps might have small maximum value. Normalize per channel to make each of them visible
    img_max, indices = torch.max(img,dim=-1,keepdim=True)
    img_max, indices = torch.max(img_max,dim=-2,keepdim=True)
    return img/img_max


if __name__ == '__main__':
    main()
