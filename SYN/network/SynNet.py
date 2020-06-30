# PyTorch includes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# Custom includes
from network.encoder import ConvEncoder
from network.encoder import FCN
from network.BiConvLSTM import BiConvLSTM
from network.classification import Classification


class SynNet(nn.Module):
    def __init__(self, clip_len):
        super(SynNet, self).__init__()
        self.clip_len = clip_len
        self.convenc = ConvEncoder()
        self.fcn = FCN()
        self.biconvlstm = BiConvLSTM(input_size=(12, 4), input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1)
        self.fc = nn.Linear(64*12*4, 256)
        self.fc2 = nn.Linear(512, 64)
        self.classification = Classification(in_size=(clip_len, 128), in_channels=clip_len, num_classes=clip_len)

        # other ways
        # self.convlstm = ConvLSTM(input_size=(14, 14), input_dim=512, hidden_dim=512, kernel_size=(3, 3), num_layers=1)
        # self.classification = Classification(in_size=(14, 14), in_channels=512, num_classes=num_classes)

    def forward(self, clips1, clips2, clip_len):
        clips1 = self.convenc(clips1.float())
        clips2 = self.convenc(clips2.float())
        clips1_fcn = self.fcn(clips1)
        clips2_fcn = self.fcn(clips2)
        clips1_fcn = clips1_fcn.reshape(clip_len, 256)
        clips2_fcn = clips2_fcn.reshape(clip_len, 256)


        clips1 = self.biconvlstm(clips1)
        clips2 = self.biconvlstm(clips2)
        clips1 = torch.squeeze(clips1)
        clips2 = torch.squeeze(clips2)
        clips1 = torch.stack([self.fc(frame.view(-1)) for frame in clips1], dim=0)
        clips2 = torch.stack([self.fc(frame.view(-1)) for frame in clips2], dim=0)



        clips1 = torch.cat((clips1,clips1_fcn), dim=1)
        clips2 = torch.cat((clips2,clips2_fcn), dim=1)
        clips1 = torch.stack([self.fc2(frame.view(-1)) for frame in clips1], dim=0)
        clips2 = torch.stack([self.fc2(frame.view(-1)) for frame in clips2], dim=0)

        # matching
        cost = torch.FloatTensor(1, clip_len, clips1.size()[0],
                                 clips1.size()[1]*2).zero_().cuda()

        for j in range(-clip_len/2, clip_len/2):
            if j == 0:
                cost[:, j+clip_len/2, :, :clips1.size()[1]] = clips1
                cost[:, j+clip_len/2, :, clips2.size()[1]:] = clips2
            elif j > 0:
                cost[:, j+clip_len/2, j:, :clips1.size()[1]] = clips1[j:, :]
                cost[:, j+clip_len/2, j:, clips2.size()[1]:] = clips2[:-j, :]
            else:
                cost[:, j+clip_len/2, :j, :clips1.size()[1]] = clips1[:j, :]
                cost[:, j+clip_len/2, :j, clips2.size()[1]:] = clips2[-j:, :]

        # Max pool :)
        classification = self.classification(cost)
        return {'classification': classification}



        # clips1 = self.convenc(clips1.float())
        # clips2 = self.convenc(clips2.float())
        #
        # clips1 = torch.stack([self.fc(frame.view(-1)) for frame in clips1], dim=0)
        # clips2 = torch.stack([self.fc(frame.view(-1)) for frame in clips2], dim=0)
        #
        #
        # # matching
        # cost = torch.FloatTensor(1, clip_len, clips1.size()[0],
        #                          clips1.size()[1]*2).zero_().cuda()
        #
        # for j in range(-clip_len/2, clip_len/2):
        #     if j == 0:
        #         cost[:, j+clip_len/2, :, :clips1.size()[1]] = clips1
        #         cost[:, j+clip_len/2, :, clips2.size()[1]:] = clips2
        #     elif j > 0:
        #         cost[:, j+clip_len/2, j:, :clips1.size()[1]] = clips1[j:, :]
        #         cost[:, j+clip_len/2, j:, clips2.size()[1]:] = clips2[:-j, :]
        #     else:
        #         cost[:, j+clip_len/2, :j, :clips1.size()[1]] = clips1[:j, :]
        #         cost[:, j+clip_len/2, :j, clips2.size()[1]:] = clips2[-j:, :]
        #
        # # Max pool :)
        # classification = self.classification(cost)
        # return {'classification': classification}