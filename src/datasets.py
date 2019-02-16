import numpy as np
import scipy.misc as misc
import os
from torch.utils.data import Dataset
import scipy


class CRCDataLoader(Dataset):
    def __init__(self, type, imgdir, task='detection', target_size=256):
        self.target_size = target_size
        assert task in ['detection', 'classification', 'joint']
        assert type in ['test', 'train', 'validation']
        self.type = type
        self.task = task
        self.imgdir = os.path.join(imgdir, type)
        self.x_train = []
        self.y_train_det = []
        self.y_train_cls = []
        self.det_mats = []
        self.load_train()

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        if self.task == 'detection':
            if self.type == 'validation' or self.type == 'test':
                return self.x_train[idx], self.y_train_det[idx], self.det_mats[idx]
            else:
                return self.x_train[idx], self.y_train_det[idx]
        elif self.task == 'classification':
            return self.x_train[idx], self.y_train_cls[idx]
        else:
            return self.x_train[idx], self.y_train_det[idx], self.y_train_cls[idx]

    def load_train(self, preprocess=True):
        for i, img_folder in enumerate(os.listdir(self.imgdir)):
            curr_img_folder = os.path.join(self.imgdir, img_folder)
            img_file_name = img_folder + '.bmp'
            det_label_name = img_folder + '_detection.bmp'
            cls_label_name = img_folder + '_classification.bmp'
            det_mat = img_folder + '_detection.mat'
            for j, file_name in enumerate(os.listdir(curr_img_folder)):
                curr_file = os.path.join(curr_img_folder, file_name)
                if file_name == img_file_name:
                    img = misc.imread(curr_file)
                    #print('img.shape: ', img.shape)
                    img = misc.imresize(img, (self.target_size, self.target_size), interp='nearest')
                    # if preprocess:
                    # img = img/255.
                    # img -= np.mean(img, keepdims=True)
                    # img /= (np.std(img, keepdims=True) + 1e-7)
                    img = img - 128.
                    img = img / 128.
                    self.x_train.append(img)
                if self.task == 'detection' or self.task == 'joint':
                    if file_name == det_label_name:
                        detection = misc.imread(curr_file, mode='L')
                        #print('detection.shape: ', detection.shape)
                        detection = misc.imresize(detection, (self.target_size, self.target_size), interp='nearest')
                        detection = detection.reshape(detection.shape[0], detection.shape[1], 1)
                        #print(detection)
                        self.y_train_det.append(detection)
                    if file_name == det_mat:
                        self.det_gt_mat = scipy.io.loadmat(curr_file)['detection']
                        self.det_gt_mat = np.uint8(self.det_gt_mat)
                        sh = self.det_gt_mat.shape[0]
                        self.det_mats.append(sh)
                if self.task == 'classification' or self.task == 'joint':
                    if file_name == cls_label_name:
                        classification = misc.imread(curr_file, mode='L')
                        #classification = misc.imresize(classification, (self.target_size, self.target_size),
                        #                               interp='nearest')
                        classification = classification.reshape(classification.shape[0], classification.shape[1], 1)
                        self.y_train_cls.append(classification)
        self.det_mats = np.array(self.det_mats)
        print(self.det_mats)
        self.x_train = np.array(self.x_train)
        self.x_train = np.transpose(self.x_train, (0, 3, 1, 2))
        if self.task == 'detection':
            self.y_train_det = np.array(self.y_train_det)
            self.y_train_det = np.transpose(self.y_train_det, (0, 3, 1, 2))
        if self.task == 'classification':
            self.y_train_cls = np.array(self.y_train_cls)
            self.y_train_cls = np.transpose(self.y_train_cls, (0, 3, 1, 2))
        if self.task == 'joint':
            self.y_train_det = np.array(self.y_train_det)
            self.y_train_det = np.transpose(self.y_train_det, (0, 3, 1, 2))
            self.y_train_cls = np.array(self.y_train_cls)
            self.y_train_cls = np.transpose(self.y_train_cls, (0, 3, 1, 2))


def _image_normalization(image, preprocess_num):
    image = image - preprocess_num
    image = image / preprocess_num
    return image


def _torch_image_transpose(images, type='image'):
    """
    :param image:
    :param type:
    :return:
    """
    images = np.array(images)
    images = np.transpose(images, (0, 3, 1, 2))
    return images
