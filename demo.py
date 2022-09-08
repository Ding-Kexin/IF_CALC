import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
from utils import setup_seed, train_2_patch_label2, show_calaError
import argparse
import os
import time
from CALC import train_network
# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("GLT")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--epoches', type=int, default=250, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Houston', help='dataset to use')
parser.add_argument('--num_classes', choices=[11, 6, 15], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=16, help='number1 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def train_1times():
    # -------------------------------------------------------------------------------
    # prepare data
    if args.dataset == 'Houston':
        DataPath1 = r'/home/s4u2/dkx_expriment/MMGAN/dataset/houston/Houston.mat'
        DataPath2 = r'/home/s4u2/dkx_expriment/MMGAN/dataset/houston/LiDAR.mat'
        LabelPath_10TIMES = r'/home/s4u2/dkx_expriment/MMGAN/dataset/houston/train_test/20/train_test_gt_1.mat'
        Data1 = scio.loadmat(DataPath1)['img']  # (349,1905,144)
        Data2 = scio.loadmat(DataPath2)['img']  # (349,1905)
    elif args.dataset == 'Muufl':
        DataPath1 = r'/home/s4u2/dkx_expriment/MMGAN/dataset/Muufl/hsi.mat'
        DataPath2 = r'/home/s4u2/dkx_expriment/MMGAN/dataset/Muufl/lidar_DEM.mat'
        Data1 = scio.loadmat(DataPath1)['hsi']
        Data2 = scio.loadmat(DataPath2)['lidar']
        LabelPath_10TIMES = r'/home/s4u2/dkx_expriment/MMGAN/dataset/Muufl/train_test/20/train_test_gt_1.mat'
    TrLabel_10TIMES = scio.loadmat(LabelPath_10TIMES)['train_data']  # 349*1905
    TsLabel_10TIMES = scio.loadmat(LabelPath_10TIMES)['test_data']  # 349*1905
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    # sample generation
    patchsize = args.patch_size  # input spatial size for 2D-CNN
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)  # 8
    TrainPatch1, TrainPatch2, TrainLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TrLabel_10TIMES)
    TestPatch1, TestPatch2, TestLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TsLabel_10TIMES)
    train_dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = dataf.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # shuffle=True!!! is key
    print('Data1 Training size and testing size are:', TrainPatch1.shape, 'and', TestPatch1.shape)
    print('Data2 Training size and testing size are:', TrainPatch2.shape, 'and', TestPatch2.shape)
    # -------------------------------------------------------------------------------
    # train and test
    tic1 = time.time()
    pred_y, val_acc = train_network(train_loader, TrainPatch1, TrainPatch2, TrainLabel,
                                    TestPatch1, TestPatch2, TestLabel, LR=args.learning_rate,
                                    EPOCH=args.epoches, patchsize=args.patch_size, l1=band1, l2=band2,
                                    Classes=args.num_classes)
    pred_y.type(torch.FloatTensor)
    TestLabel.type(torch.FloatTensor)
    # pred_y_out = np.array(pred_y)
    # TestLabel_out = np.array(TestLabel)
    print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
    toc1 = time.time()
    time_1 = toc1 - tic1
    print('1st training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    OA, Kappa, CA, AA = show_calaError(pred_y, TestLabel)
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))


if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()
