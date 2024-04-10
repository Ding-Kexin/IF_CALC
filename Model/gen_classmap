import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
from utils import setup_seed, train_2_patch_label2
import sys
import time
from CALC import Network
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# ######load dataset######
DataPath1 = r'/home/server04/dkx/dkx_experiment/dataset/Muufl/hsi.mat'
DataPath2 = r'/home/server04/dkx/dkx_experiment/dataset/Muufl/lidar_DEM.mat'
Data1 = scio.loadmat(DataPath1)['hsi']
Data2 = scio.loadmat(DataPath2)['lidar']
# ######parameter settings######
patchsize = 16  # input spatial size for 2D-CNN
batchsize = 64  # select from [16, 32, 64, 128], the best is 64
TrainNum = 300
EPOCH = 250
LR = 0.001
pad_width = np.floor(patchsize / 2)
pad_width = int(pad_width)  # 8
[m1, n1, l1] = np.shape(Data1)
# Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
[m2, n2, l2] = np.shape(Data2)
Classes = 11
MODEL = 'WHOLE'
# ######data processing######
Data1 = Data1.astype(np.float32)
Data2 = Data2.astype(np.float32)
def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0
if MODEL.strip() == 'WHOLE':
    setup_seed(seed=1)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1  # 349*1905*144
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2  # 349*1905*21

    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')  # 365*1921*144
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')  # 365*1921*21

    for i in range(l1):
        temp = x1[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x1_pad[:, :, i] = temp2  # 365*1921*144
    for i in range(l2):
        temp = x2[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x2_pad[:, :, i] = temp2  # 365*1921*21

    cnn = Network(l1=l1, l2=l2, Classes=Classes)
    cnn.cuda()
    cnn.load_state_dict(torch.load('/home/server04/dkx/dkx_experiment/MMGAN/code/for_git/CALC_Muufl.pkl'))
    cnn.eval()
    # w0 = w0B
    # w1 = w1B
    # w2 = w2B
    # show the whole image
    # The whole data is too big to test in one time; So dividing it into several parts
    part = 100
    pred_all = np.ones((m1 * n1, 1), dtype='float32')

    number = m1 * n1 // part
    for i in range(number):
        D1 = np.empty((part, l1, patchsize, patchsize), dtype='float32')
        D2 = np.empty((part, l2, patchsize, patchsize), dtype='float32')
        count = 0
        for j in range(i * part, (i + 1) * part):
            row = j // n1
            col = j - row * n1
            row2 = row + pad_width
            col2 = col + pad_width
            patch1 = x1_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch2 = x2_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch1 = np.reshape(patch1, (patchsize * patchsize, l1))
            patch2 = np.reshape(patch2, (patchsize * patchsize, l2))
            patch1 = np.transpose(patch1)
            patch2 = np.transpose(patch2)
            patch1 = np.reshape(patch1, (l1, patchsize, patchsize))
            patch2 = np.reshape(patch2, (l2, patchsize, patchsize))
            D1[count, :, :, :] = patch1
            D2[count, :, :, :] = patch2
            count = count + 1
            # count += 1

        temp1 = torch.from_numpy(D1)
        temp1_2 = torch.from_numpy(D2)
        temp1 = temp1.cuda()
        temp1_2 = temp1_2.cuda()
        _,_, temp2, _, _ = cnn(temp1,temp1_2)
        # temp2 = w2 * cnn(temp1_1, temp1_2)[4] + w1 * cnn(temp1_1, temp1_2)[3] + w0 * cnn(temp1_1, temp1_2)[
        #     2]
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[i * part:(i + 1) * part, 0] = temp3.cpu()
        del temp1, temp1_2, _, temp2, temp3, D1, D2

    if (i + 1) * part < m1 * n1:
        D1 = np.empty((m1 * n1 - (i + 1) * part, l1, patchsize, patchsize), dtype='float32')
        D2 = np.empty((m2 * n2 - (i + 1) * part, l2, patchsize, patchsize), dtype='float32')
        count = 0
        for j in range((i + 1) * part, m1 * n1):
            row = j // n1
            col = j - row * n1
            row2 = row + pad_width
            col2 = col + pad_width
            patch1 = x1_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch2 = x2_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch1 = np.reshape(patch1, (patchsize * patchsize, l1))
            patch1 = np.transpose(patch1)
            patch2 = np.reshape(patch2, (patchsize * patchsize, l2))
            patch2 = np.transpose(patch2)
            patch1 = np.reshape(patch1, (l1, patchsize, patchsize))
            patch2 = np.reshape(patch2, (l2, patchsize, patchsize))
            D1[count, :, :, :] = patch1
            D2[count, :, :, :] = patch2
            count = count + 1
            # count += 1

        temp1 = torch.from_numpy(D1)
        temp1_2 = torch.from_numpy(D2)
        temp1 = temp1.cuda()
        temp1_2 = temp1_2.cuda()
        _,_, temp2, _, _ = cnn(temp1, temp1_2)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[(i + 1) * part:m1 * n1, 0] = temp3.cpu()
        del temp1, temp1_2, _, temp2, temp3, D1, D2

    pred_all = np.reshape(pred_all, (m1, n1)) + 1
    # OA = OA.numpy()
    # pred_all = pred_all.cpu()
    # pred_all = pred_all.numpy()
    # TestDataLabel = TestLabel.cpu()
    # TestDataLabel = TestDataLabel.numpy()

    scio.savemat(r'./muufl_pred_all_new2.mat',
                 {'pred_all': pred_all})
    ###################################################
    best_G = pred_all
    hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
    for i in range(best_G.shape[0]):
        for j in range(best_G.shape[1]):
            if best_G[i][j] == 0:
                hsi_pic[i, j, :] = [0, 0, 0]
            if best_G[i][j] == 1:
                hsi_pic[i, j, :] = [0, 0, 1]
            if best_G[i][j] == 2:
                hsi_pic[i, j, :] = [0, 1, 0]
            if best_G[i][j] == 3:
                hsi_pic[i, j, :] = [0, 1, 1]
            if best_G[i][j] == 4:
                hsi_pic[i, j, :] = [1, 0, 0]
            if best_G[i][j] == 5:
                hsi_pic[i, j, :] = [1, 0, 1]
            if best_G[i][j] == 6:
                hsi_pic[i, j, :] = [1, 1, 0]
            if best_G[i][j] == 7:
                hsi_pic[i, j, :] = [0.5, 0.5, 1]
            if best_G[i][j] == 8:
                hsi_pic[i, j, :] = [1, 0.2, 0.5]
            if best_G[i][j] == 9:
                hsi_pic[i, j, :] = [0.3, 0.4, 1]
            if best_G[i][j] == 10:
                hsi_pic[i, j, :] = [1, 1, 0.6]
            if best_G[i][j] == 11:
                hsi_pic[i, j, :] = [0.1, 0.5, 1]

    classification_map(hsi_pic[2:-2, 2:-2, :], best_G[2:-2, 2:-2], 24, "./muufl_classmap2.png")



