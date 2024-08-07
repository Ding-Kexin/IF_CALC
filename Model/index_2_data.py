import numpy as np
import scipy.io as scio

def samplingFixedNum(sample_num, groundTruth, seed):              #divide dataset into train and test datasets
    labels_loc = {}
    train_ = {}
    test_ = {}
    np.random.seed(seed)
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        train_[i] = indices[:sample_num]
        test_[i] = indices[sample_num:]                     #difference derivation
    train_fix_indices = []
    test_fix_indices = []
    for i in range(m):
        train_fix_indices += train_[i]
        test_fix_indices += test_[i]
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)
    return train_fix_indices, test_fix_indices

train_num = 20
for i in range(10):
    seed = i + 1
    Muufl_gt = scio.loadmat(./dataset/Muufl/Muufl_gt.mat')
    Muufl_gt = Muufl_gt['Muufl_gt']
    gt = Muufl_gt.reshape(np.prod(Muufl_gt.shape[:2]), ).astype(np.int)
    train_index, test_index = samplingFixedNum(train_num, gt, seed)
    train_data = np.zeros(np.prod(Muufl_gt.shape[:2]), )
    train_data[train_index] = gt[train_index]
    test_data = np.zeros(np.prod(Muufl_gt.shape[:2]), )
    test_data[test_index] = gt[test_index]
    train_data = train_data.reshape(np.prod(Muufl_gt.shape[:1]),np.prod(Muufl_gt.shape[1:]) )
    test_data = test_data.reshape(np.prod(Muufl_gt.shape[:1]), np.prod(Muufl_gt.shape[1:]))
    scio.savemat(r'./train_test/%d/train_test_gt_%d.mat'%(train_num, i+1),
                {'train_data': train_data, 'test_data': test_data,
                 'train_index': train_index, 'test_index': test_index})
