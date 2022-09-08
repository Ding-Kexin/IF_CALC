# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 15:10
# @Author  : DingKexin
# @FileName: utils.py
# @Software: PyCharm
import torch
import numpy as np

def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = torch.sum(predict==label)*1.0/n
    correct_sum = torch.zeros((max(label)+1))
    reali = torch.zeros((max(label)+1))
    predicti = torch.zeros((max(label)+1))
    CA = torch.zeros((max(label)+1))
    for i in range(0, max(label) + 1):
        correct_sum[i] = torch.sum(label[np.where(predict == i)] == i)
        reali[i] = torch.sum(label == i)
        predicti[i] = torch.sum(predict == i)
        CA[i] = correct_sum[i] / reali[i]

    Kappa = (n * torch.sum(correct_sum) - torch.sum(reali * predicti)) * 1.0 / (n * n - torch.sum(reali * predicti))
    AA = torch.mean(CA)
    return OA, Kappa, CA, AA


def show_calaError(val_predict_labels, val_true_labels):
   val_predict_labels = torch.squeeze(val_predict_labels)
   val_true_labels = torch.squeeze(val_true_labels)
   OA, Kappa, CA, AA = CalAccuracy(val_predict_labels, val_true_labels)
   # ic(OA, Kappa, CA, AA)
   print("OA: %f, Kappa: %f,  AA: %f" % (OA, Kappa, AA))
   print("CA: ",)
   print(CA)
   return OA, Kappa, CA, AA


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_2_patch_label2(Data1, Data2, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    [m2, n2, l2] = np.shape(Data2)
    # [m3, n3, l3] = np.shape(Data3)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1  # 349*1905*144
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2  # 349*1905*21
    # for i in range(l3):
    #     Data3[:, :, i] = (Data3[:, :, i] - Data3[:, :, i].min()) / (Data3[:, :, i].max() - Data3[:, :, i].min())
    # x3 = Data3  # 349*1905*21
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')  # 365*1921*144
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')  # 365*1921*21
    # x3_pad = np.empty((m3 + patchsize, n3 + patchsize, l3), dtype='float32')  # 365*1921*21
    for i in range(l1):
        temp = x1[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x1_pad[:, :, i] = temp2  # 365*1921*144
    for i in range(l2):
        temp = x2[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x2_pad[:, :, i] = temp2  # 365*1921*21
    # for i in range(l3):
    #     temp = x3[:, :, i]  # 349*1905
    #     temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
    #     x3_pad[:, :, i] = temp2  # 365*1921*21
    # construct the training and testing set
    [ind1, ind2] = np.where(Label > 0)  # [300,300]  !=0  change > 0 ,muufl = -1
    TrainNum = len(ind1)  # 300
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')  # 300*144**16*16
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')  # 300*21**16*16
    # TrainPatch3 = np.empty((TrainNum, l3, patchsize, patchsize), dtype='float32')  # 300*144**16*16
    TrainLabel = np.empty(TrainNum)  # 300
    ind3 = ind1 + pad_width  # 300
    ind4 = ind2 + pad_width  # 300
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # 16*16*144
        patch1 = np.transpose(patch1, (2, 0, 1))  # 144*16*16
        TrainPatch1[i, :, :, :] = patch1  # 300*144*16*16
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # 16*16*21
        patch2 = np.transpose(patch2, (2, 0, 1))  # 21*16*16
        TrainPatch2[i, :, :, :] = patch2  # 300*21*16*16
        # patch3 = x3_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width),
        #          :]  # 16*16*21
        # patch3 = np.transpose(patch3, (2, 0, 1))  # 21*16*16
        # TrainPatch3[i, :, :, :] = patch3  # 300*21*16*16
        patchlabel = Label[ind1[i], ind2[i]]  # 1
        TrainLabel[i] = patchlabel  # 300
    # step3: change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    # TrainPatch3 = torch.from_numpy(TrainPatch3)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainLabel

