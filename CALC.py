# -*- coding:utf-8 -*-
# @Time       :2022/9/8 上午11:07
# @AUTHOR     :DingKexin
# @FileName   :CALC.py
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=l1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # add pool
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=l2,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # add pool
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(2),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # add pool
            # nn.Dropout(0.5),

        )
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

    def forward(self, x1, x2):
        x1_1 = self.conv1_1(x1)  # 64*32*16*16
        x1_2 = self.conv1_2(x2)  # 64*32*16*16
        x1_add = x1_1 * self.xishu1 + x1_2 * self.xishu2
        x2_1 = self.conv2_1(x1_1)  # 64*64*8*8
        x2_2 = self.conv2_1(x1_2)  # 64*64*8*8
        x2_add = x2_1 * self.xishu1 + x2_2 * self.xishu2
        x3_1 = self.conv3_1(x2_1)  # 64*128*8*8
        x3_2 = self.conv3_1(x2_2)  # 64*128*8*8
        x3_add = x3_1 * self.xishu1 + x3_2 * self.xishu2
        return x1_add, x2_add, x3_add


class Decoder(nn.Module):
    def __init__(self, l1, l2):
        super(Decoder, self).__init__()

        self.dconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.dconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )

        self.dconv3_H = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(32, l1, 3, 1, 1),
            # nn.BatchNorm2d(l1),           #      qudiao
            nn.Sigmoid(),

        )
        self.dconv3_L = nn.Sequential(
            nn.Upsample(scale_factor=2),   # add Upsample
            nn.Conv2d(32, l2, 3, 1, 1),
            # nn.BatchNorm2d(l1),           #      qudiao
            nn.Sigmoid(),

        )

    def forward(self, x1_cat):
        x = self.dconv1(x1_cat)
        x = self.dconv2(x)
        x_H = self.dconv3_H(x)
        x_L = self.dconv3_L(x)
        return x_H, x_L


class Classifier(nn.Module):
    def __init__(self, Classes):
        super(Classifier, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(16, Classes, 1),
        )
        # self.out1 = nn.Linear(32, Classes)  # fully connected layer, output 16 classes
        # self.out2 = nn.Linear(32, Classes)
        # self.out3 = nn.Linear(32, Classes)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.31]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.33]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([0.36]))
        # self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.14]))
        # self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.29]))
        # self.coefficient3 = torch.nn.Parameter(torch.Tensor([0.57]))

    def forward(self, x1, x2, x3):
        x1_1 = self.conv1(x1)  # 64*256*8*8 -> 64*128*8*8
        x1_2 = self.conv2(x1_1)  # 64*128*8*8 -> 64*64*1*1
        x1_3 = self.conv3(x1_2)  # 64*64*1*1 -> 64*15*1*1
        x1_3 = x1_3.view(x1_3.size(0), -1)  # 64*15
        x1_out = F.softmax(x1_3, dim=1)

        x2_1 = self.conv1_2(x2)  # 64*128*8*8 -> 64*64*8*8
        x2_2 = self.conv2_2(x2_1)  # 64*64*8*8 -> 64*32*1*1
        x2_3 = self.conv3_2(x2_2)  # 64*32*1*1 -> 64*15*1*1
        x2_3 = x2_3.view(x2_3.size(0), -1)  # 64*15
        x2_out = F.softmax(x2_3, dim=1)

        x3_1 = self.conv1_3(x3)  # 64*64*16*16 -> 64*32*16*16
        x3_2 = self.conv2_3(x3_1)  # 64*32*16*16 -> 64*16*1*1
        x3_3 = self.conv3_3(x3_2)  # 64*16*1*1 -> 64*15*1*1
        x3_3 = x3_3.view(x3_3.size(0), -1)  # 64*15
        x3_out = F.softmax(x3_3, dim=1)

        return x1_out, x2_out, x3_out


class Discriminator_H(nn.Module):
    def __init__(self, l1):
        super(Discriminator_H, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 15)
        self.sa = SpatialAttention()

    def forward(self, input):  # input:64*144*16*16
        input = self.sa(input) * input
        x1 = self.conv1(input)  # 64*32*16*16
        x2 = self.conv2(x1)  # 64*32*8*8
        x3 = self.conv3(x2)  # 64*64*8*8
        x4 = self.conv4(x3)  # 64*64*4*4
        x5 = self.conv5(x4)  # 64*128*4*4
        x6 = self.conv6(x5)  # 64*128*2*2
        x7 = self.avgpool(x6)  # 64*128*1*1
        x8 = x7.view(x7.size(0), -1)  # 64*128
        x9 = self.fc(x8)  # 64*1
        x10 = torch.sigmoid(x9)  # 64*1
        return x10


class Discriminator_L(nn.Module):
    def __init__(self, l2):
        super(Discriminator_L, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(l2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 15)
        self.sa = SpatialAttention()

    def forward(self, input):  # input:64*144*16*16
        input = self.sa(input) * input
        x1 = self.conv1(input)  # 64*32*16*16
        x2 = self.conv2(x1)  # 64*32*8*8
        x3 = self.conv3(x2)  # 64*64*8*8
        x4 = self.conv4(x3)  # 64*64*4*4
        x5 = self.conv5(x4)  # 64*128*4*4
        x6 = self.conv6(x5)  # 64*128*2*2
        x7 = self.avgpool(x6)  # 64*128*1*1
        x8 = x7.view(x7.size(0), -1)  # 64*128
        x9 = self.fc(x8)  # 64*1
        x10 = torch.sigmoid(x9)  # 64*1
        return x10


class Network(nn.Module):
    def __init__(self, l1, l2, Classes):
        super(Network, self).__init__()
        self.encoder = Encoder(l1=l1, l2=l2)
        self.decoder = Decoder(l1=l1, l2=l2)
        self.classifier = Classifier(Classes=Classes)

    def forward(self, x1, x2):
        ex1, ex2, ex3 = self.encoder(x1, x2)
        rx_H, rx_L = self.decoder(ex3)
        cx1, cx2, cx3 = self.classifier(ex3, ex2, ex1)
        rx_H = rx_H.view(rx_H.size(0), -1)
        rx_L = rx_L.view(rx_L.size(0), -1)
        return rx_H, rx_L, cx1, cx2, cx3


# train and test the designed model

def train_network(train_loader, TrainPatch1, TrainPatch2, TrainLabel1, TestPatch1, TestPatch2, TestLabel, LR, EPOCH, patchsize, l1, l2, Classes):
    cnn = Network(l1=l1, l2=l2, Classes=Classes)
    dis_H = Discriminator_H(l1=l1)
    dis_L= Discriminator_L(l2=l2)
    cnn.cuda()
    dis_H.cuda()
    dis_L.cuda()
    g_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    d_optimizer_H = torch.optim.Adam(dis_H.parameters(), lr=LR)
    d_optimizer_L = torch.optim.Adam(dis_L.parameters(), lr=LR)
    loss_fun1 = nn.CrossEntropyLoss()  # the target label is not one-hotted
    loss_fun2 = nn.MSELoss()
    val_acc = []
    class_loss = []
    gan_loss = []
    BestAcc = 0
    for epoch in range(EPOCH):
        for step, (b_x1, b_x2, b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            # move train data to GPU
            b_x1 = b_x1.cuda()  # 64*144*16*16
            b_x2 = b_x2.cuda()  # 64*21*16*16
            b_y = b_y.cuda()  # 64
            fake_H, fake_L, output, output2, output3 = cnn(b_x1, b_x2)  # fake_img:64*(144*16*16) output:64*15    # cnn output
            dis_H.zero_grad()
            fake_probability = dis_H(fake_H.view(fake_H.size(0), l1, patchsize, patchsize))
            fake_probability = fake_probability.mean()
            real_probability = dis_H(b_x1)
            real_probability = real_probability.mean()
            d_loss1 = 1 - real_probability + fake_probability
            d_loss1.backward(retain_graph=True)
            dis_L.zero_grad()
            fake_probability2 = dis_L(fake_L.view(fake_L.size(0), l2, patchsize, patchsize))
            fake_probability2 = fake_probability2.mean()
            real_probability2 = dis_L(b_x2)
            real_probability2 = real_probability2.mean()
            d_loss2 = 1 - real_probability2 + fake_probability2  # -1 - real + fake
            d_loss2.backward(retain_graph=True)
            cnn.zero_grad()
            ce_loss = loss_fun1(output, b_y.long()) + loss_fun2(output, output2) + loss_fun2(output, output3)
            a_loss = 0.5 * torch.mean(1 - fake_probability) + 0.5 * torch.mean(1 - fake_probability2)
            g_loss = 0.01 * a_loss + ce_loss
            g_loss.backward()
            d_optimizer_H.step()
            d_optimizer_L.step()
            g_optimizer.step()

            if step % 50 == 0:
                cnn.eval()
                temp1 = TrainPatch1
                temp1 = temp1.cuda()
                temp2 = TrainPatch2
                temp2 = temp2.cuda()
                _, _, temp3, temp4, temp5 = cnn(temp1, temp2)
                pred_y1 = torch.max(temp3, 1)[1].squeeze()
                pred_y1 = pred_y1.cpu()
                acc1 = torch.sum(pred_y1 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)
                pred_y2 = torch.max(temp4, 1)[1].squeeze()
                pred_y2 = pred_y2.cpu()
                acc2 = torch.sum(pred_y2 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)
                pred_y3 = torch.max(temp5, 1)[1].squeeze()
                pred_y3 = pred_y3.cpu()
                acc3 = torch.sum(pred_y3 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)
                # weights are determined by each class accuracy
                Classes = np.unique(TrainLabel1)
                w0 = np.empty(len(Classes), dtype='float32')
                w1 = np.empty(len(Classes), dtype='float32')
                w2 = np.empty(len(Classes), dtype='float32')
                for i in range(len(Classes)):
                    cla = Classes[i]
                    right1 = 0
                    right2 = 0
                    right3 = 0

                    for j in range(len(TrainLabel1)):
                        if TrainLabel1[j] == cla and pred_y1[j] == cla:
                            right1 += 1
                        if TrainLabel1[j] == cla and pred_y2[j] == cla:
                            right2 += 1
                        if TrainLabel1[j] == cla and pred_y3[j] == cla:
                            right3 += 1

                    w0[i] = right1.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                    w1[i] = right2.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                    w2[i] = right3.__float__() / (right1 + right2 + right3 + 0.00001).__float__()

                w0 = torch.from_numpy(w0).cuda()
                w1 = torch.from_numpy(w1).cuda()
                w2 = torch.from_numpy(w2).cuda()

                pred_y = np.empty((len(TestLabel)), dtype='float32')
                number = len(TestLabel) // 100
                for i in range(number):
                    temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :]
                    temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :]
                    temp1_1 = temp1_1.cuda()
                    temp1_2 = temp1_2.cuda()
                    temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[
                        2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
                    del temp1_1, temp1_2, temp2, temp3

                if (i + 1) * 100 < len(TestLabel):
                    temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :]
                    temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :]
                    temp1_1 = temp1_1.cuda()
                    temp1_2 = temp1_2.cuda()
                    temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[
                        2]
                    temp3 = torch.max(temp2, 1)[1].squeeze()
                    pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
                    del temp1_1, temp1_2, temp2, temp3

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                print('Epoch: ', epoch, '| classify loss: %.6f' % ce_loss.data.cpu().numpy(),
                      '| test accuracy: %.6f' % accuracy, '| w0: %.2f' % w0[0], '| w1: %.2f' % w1[0],
                      '| w2: %.2f' % w2[0])
                val_acc.append(accuracy.data.cpu().numpy())
                class_loss.append(ce_loss.data.cpu().numpy())
                gan_loss.append(g_loss.data.cpu().numpy())
                # save the parameters in network
                if accuracy > BestAcc:
                    torch.save(cnn.state_dict(), 'CALC.pkl')
                    BestAcc = accuracy
                    w0B = w0
                    w1B = w1
                    w2B = w2
                cnn.train()  ### 启用 Batch Normalization 和 Dropout

    cnn.load_state_dict(torch.load('CALC.pkl'))
    cnn.eval()
    w0 = w0B
    w1 = w1B
    w2 = w2B

    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel) // 100
    for i in range(number):
        temp1_1 = TestPatch1[i * 100:(i + 1) * 100, :, :, :]
        temp1_2 = TestPatch2[i * 100:(i + 1) * 100, :, :, :]
        temp1_1 = temp1_1.cuda()
        temp1_2 = temp1_2.cuda()
        temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[2]
        temp3 = torch.max(temp2, 1)[1].squeeze()

        pred_y[i * 100:(i + 1) * 100] = temp3.cpu()
        del temp1_1, temp1_2, temp2, temp3

    if (i + 1) * 100 < len(TestLabel):
        temp1_1 = TestPatch1[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_2 = TestPatch2[(i + 1) * 100:len(TestLabel), :, :, :]
        temp1_1 = temp1_1.cuda()
        temp1_2 = temp1_2.cuda()
        temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[2]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * 100:len(TestLabel)] = temp3.cpu()
        del temp1_1, temp1_2, temp2, temp3

    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

    Classes = np.unique(TestLabel)
    EachAcc = np.empty(len(Classes))

    for i in range(len(Classes)):
        cla = Classes[i]
        right = 0
        sum = 0

        for j in range(len(TestLabel)):
            if TestLabel[j] == cla:
                sum = sum + 1
                # sum += 1

            if TestLabel[j] == cla and pred_y[j] == cla:
                right = right + 1
                # right += 1

        EachAcc[i] = right.__float__() / sum.__float__()
        AA = np.mean(EachAcc)

    print(OA)
    print(EachAcc)
    print(AA)
    return pred_y, val_acc



