import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
np.set_printoptions(threshold=np.inf)
import os, cv2, shutil
from scipy.io import loadmat
from PIL import Image, ImageDraw
from glob import glob
#from imgaug import augmenters as iaa
#import imgaug as ia
#from config import Config
import torchvision.transforms as transforms


ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data', 'cls_and_det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')


def _isArrayLike(obj):
    """
    check if this is array like object.
    """
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

"""
def aug_on_fly(img, det_mask, cls_mask):


    def image_basic_augmentation(image, masks, ratio_operations=0.9):
        # without additional operations
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            sometimes(
                iaa.SomeOf((0, 5), [
                    iaa.Fliplr(hor_flip_angle),
                    iaa.Flipud(ver_flip_angle),
                    iaa.Affine(shear=(-16, 16)),
                    iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                    iaa.PerspectiveTransform(scale=(0.01, 0.1))
                ]))
        ])
        det_mask, cls_mask = masks[0], masks[1]
        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_images(image)
        aug_det_mask = seq_to_deterministic.augment_images(det_mask)
        aug_cls_mask = seq_to_deterministic.augment_images(cls_mask)
        return aug_img, aug_det_mask, aug_cls_mask

    aug_image, aug_det_mask, aug_cls_mask = image_basic_augmentation(image=img, masks=[det_mask, cls_mask])
    return aug_image, aug_det_mask, aug_cls_mask


def heavy_aug_on_fly(img, det_mask):
    def image_heavy_augmentation(image, det_masks, ratio_operations=0.6):
        # according to the paper, operations such as shearing, fliping horizontal/vertical,
        # rotating, zooming and channel shifting will be apply
        sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)
        edge_detect_sometime = lambda aug: iaa.Sometimes(0.1, aug)
        elasitic_sometime = lambda aug: iaa.Sometimes(0.2, aug)
        add_gauss_noise = lambda aug: iaa.Sometimes(0.15, aug)
        hor_flip_angle = np.random.uniform(0, 1)
        ver_flip_angle = np.random.uniform(0, 1)
        seq = iaa.Sequential([
            iaa.SomeOf((0, 5), [
                iaa.Fliplr(hor_flip_angle),
                iaa.Flipud(ver_flip_angle),
                iaa.Affine(shear=(-16, 16)),
                iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
                iaa.PerspectiveTransform(scale=(0.01, 0.1)),

                # These are additional augmentation.
                # iaa.ContrastNormalization((0.75, 1.5))

            ])])
        # elasitic_sometime(
        seq_to_deterministic = seq.to_deterministic()
        aug_img = seq_to_deterministic.augment_images(image)
        aug_det_mask = seq_to_deterministic.augment_images(det_masks)
        return aug_img, aug_det_mask

    aug_image, aug_det_mask = image_heavy_augmentation(image=img, det_masks=det_mask)
    return aug_image, aug_det_mask

"""
def train_test_split(data_path, notation_type, new_folder='cls_and_det',
                     test_sample=20, valid_sample=10):
    """
    randomly split data into train, test and validation set. pre-defined number was based
    on sfcn-opi paper
    :param data_path: main path for the original data. ie: ../data
    :param new_folder: new folder to store train, test, and validation files
    :param train_sample: number of train sample
    :param test_sample: number of test sample
    :param valid_sample: number of validation sample
    """
    if notation_type == 'ellipse':
        new_folder_path = os.path.join(data_path, new_folder + '_ellipse')
    elif notation_type == 'point':
        new_folder_path = os.path.join(data_path, new_folder + '_point')
    else:
        raise Exception('notation type needs to be either ellipse or point')

    train_new_folder = os.path.join(new_folder_path, 'train')
    test_new_folder = os.path.join(new_folder_path, 'test')
    valid_new_folder = os.path.join(new_folder_path, 'validation')
    check_folder_list = [new_folder_path, train_new_folder, test_new_folder, valid_new_folder]
    check_directory(check_folder_list)

    detection_folder = os.path.join(data_path, 'Detection')
    classification_folder = os.path.join(data_path, 'Classification')

    # Wrong if number of images in detection and classification folder are not match.
    # assert len(os.listdir(detection_folder)) == len(os.listdir(classification_folder))
    length = len(os.listdir(detection_folder))

    image_order = np.arange(1, length + 1)
    np.random.shuffle(image_order)

    for i, order in enumerate(image_order):
        img_folder = os.path.join(classification_folder, 'img{}'.format(order))
        det_mat = os.path.join(detection_folder, 'img{}'.format(order), 'img{}_detection.mat'.format(order))
        if i < test_sample:
            shutil.move(img_folder, test_new_folder)
            new = os.path.join(test_new_folder, 'img{}'.format(order))
            shutil.move(det_mat, new)
        elif i < test_sample + valid_sample:
            shutil.move(img_folder, valid_new_folder)
            new = os.path.join(valid_new_folder, 'img{}'.format(order))
            shutil.move(det_mat, new)
        else:
            shutil.move(img_folder, train_new_folder)
            new = os.path.join(train_new_folder, 'img{}'.format(order))
            shutil.move(det_mat, new)
        mats = glob('{}/*.mat'.format(new), recursive=True)
        mat_list = []

        for mat in mats:
            store_name = mat.split('.')[0]
            mat_content = loadmat(mat)
            img = Image.open(os.path.join(new, 'img{}.bmp'.format(order)))
            img.save(os.path.join(new, 'img{}_original.bmp'.format(order)))

            if 'detection' in store_name:
                mask = _create_binary_masks_ellipse(mat_content, notation_type=notation_type, usage='Detection')
                mask.save('{}.bmp'.format(store_name))
                verify_img = _drawdots_on_origin_image(mat_content, notation_type=notation_type, usage='Detection',
                                                       img=img)
                verify_img.save('{}/img{}_verify_det.bmp'.format(new, order))
            elif 'detection' not in store_name:
                mat_list.append(mat_content)
        # if order == 1:
        #   print(mat_list)
        cls_mask = _create_binary_masks_ellipse(mat_list, notation_type=notation_type, usage='Classification')
        cls_mask.save('{}/img{}_classification.bmp'.format(new, order))
        verify_img = _drawdots_on_origin_image(mat_list, usage='Classification', notation_type=notation_type, img=img)
        verify_img.save('{}/img{}_verify_cls.bmp'.format(new, order))

    # _reorder_image_files(new_folder_path)


def _reorder_image_files(datapath, files=['train', 'test', 'validation']):
    for file in files:
        sub_path = os.path.join(datapath, file)
        for i, img_folder in enumerate(os.listdir(sub_path)):
            new_img_folder = os.path.join(sub_path, 'img{}'.format(i + 1))
            shutil.move(os.path.join(sub_path, img_folder), new_img_folder)
            dir = [os.path.join(new_img_folder, img_file) for img_file in os.listdir(new_img_folder)]
            for d in dir:
                print('this is d: ', d)
                # pattern = re.split('img[\d]+', )
                # match = pattern.match(d)
                # print('match: ', pattern)
                start = d.find('/img\d_')
                new_file = os.path.join(img_folder, 'img{}'.format(i + 1), d[start:])
                os.rename(d, new_file)


def check_directory(file_path):
    """
    make new file on path if file is not already exist.
    :param file_path: file_path can be list of files to create.
    """
    if _isArrayLike(file_path):
        for file in file_path:
            if not os.path.exists(file):
                os.makedirs(file)
    else:
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def check_cv2_imwrite(file_path, file):
    if not os.path.exists(file_path):
        cv2.imwrite(file_path, file)


def _draw_points(dots, img, color, notation_type, radius=3):
    if dots is not None:
        canvas = ImageDraw.Draw(img)
        if notation_type == 'point':
            for i, dot in enumerate(dots):
                canvas.point(dot, fill=color)
        elif notation_type == 'ellipse':
            for i in range(len(dots)):
                x0 = dots[i, 0] - radius
                y0 = dots[i, 1] - radius
                x1 = dots[i, 0] + radius
                y1 = dots[i, 1] + radius
                canvas.ellipse((x0, y0, x1, y1), fill=color)


def _create_binary_masks_ellipse(mats, usage, notation_type, color=[1, 2, 3, 4]):
    """
    create binary mask using loaded data
    :param mats: points, mat format
    :param usage: Detection or Classfifcation
    :param notation_type: For now, either ellipse or point
    :return: mask
    """
    mask = Image.new('L', (500, 500), 0)
    if usage == 'Classification':
        for i, mat in enumerate(mats):
            mat_content = mat['detection']
            if notation_type == 'ellipse':
                _draw_points(mat_content, mask, notation_type=notation_type, color=color[i])
            elif notation_type == 'point':
                _draw_points(mat_content, mask, color=color[i], notation_type=notation_type)
    elif usage == 'Detection':
        mat_content = mats['detection']
        if notation_type == 'ellipse':
            _draw_points(mat_content, mask, color=1, notation_type=notation_type)
        elif notation_type == 'point':
            _draw_points(mat_content, mask, color=1, notation_type=notation_type)
    return mask


def _drawdots_on_origin_image(mats, usage, img, notation_type, color=['yellow', 'green', 'blue', 'red']):
    """
    For visualizatoin purpose, draw different color on original image.
    :param mats:
    :param usage: Detection or Classfifcation
    :param img: original image
    :param color: color list for each category
    :return: dotted image
    """
    if usage == 'Classification':
        for i, mat in enumerate(mats):
            mat_content = mat['detection']
            _draw_points(mat_content, img, color[i], notation_type=notation_type)
    elif usage == 'Detection':
        mat_content = mats['detection']
        _draw_points(mat_content, img, color[0], notation_type=notation_type)
    return img


def create_binary_masks(mat):
    polygon = [(point[0], point[1]) for point in mat]
    # print(polygon)
    mask = Image.new('L', (500, 500), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    return mask
    # cv2.imshow('img', np.array(mask))
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    #


# mask.save('/home/yichen/Desktop/sfcn-opi-yichen/data/cls_and_det/validation/img4/img4_mask.png')
# print(mask)
# return mask

"""
def img_test(i, type):

   # visiualize certain image by showing all corresponding images.
   # :param i: which image
   # :param type: train, test or validation

    img = Image.open(os.path.join(p, 'cls_and_det', type, 'img{}'.format(i), 'img{}.bmp'.format(i)))
    imgd = Image.open(
        os.path.join(p, 'cls_and_det', type, 'img{}'.format(i), 'img{}_detection.bmp'.format(i)))
    imgc = Image.open(
        os.path.join(p, 'cls_and_det', type, 'img{}'.format(i), 'img{}_classification.bmp'.format(i)))
    imgv = Image.open(
        os.path.join(p, 'cls_and_det', type, 'img{}'.format(i), 'img{}_verifiy_classification.bmp'.format(i)))
    imgz = Image.open(
        os.path.join(p, 'cls_and_det', type, 'img{}'.format(i), 'img{}_verifiy_detection.bmp'.format(i)))
    contrast = ImageEnhance.Contrast(imgd)
    contrast2 = ImageEnhance.Contrast(imgc)
    img.show(img)
    imgv.show(imgv)
    imgz.show(imgz)
    contrast.enhance(20).show(imgd)
    contrast2.enhance(20).show(imgc)

"""
def load_data(data_path, type, det=True, cls=False, reshape_size=None):
    path = os.path.join(data_path, type)  # cls_and_det/train
    imgs, det_masks, cls_masks = [], [], []
    for i, file in enumerate(os.listdir(path)):
        for j, img_file in enumerate(os.listdir(os.path.join(path, file))):
            if 'original.bmp' in img_file:
                img_path = os.path.join(path, file, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if reshape_size is not None:
                    img = cv2.resize(img, reshape_size)
                img = _image_normalization(img)
                imgs.append(img)
            elif 'detection.bmp' in img_file and det == True:
                det_mask_path = os.path.join(path, file, img_file)
                # det_mask = skimage.io.imread(det_mask_path, True).astype(np.bool)
                det_mask = cv2.imread(det_mask_path, 0)
                if reshape_size is not None:
                    det_mask = cv2.resize(det_mask, reshape_size)
                det_masks.append(det_mask)
            elif 'classification.bmp' in img_file and cls == True:
                if cls == True:
                    cls_mask_path = os.path.join(path, file, img_file)
                    cls_mask = cv2.imread(cls_mask_path, 0)
                    if reshape_size != None:
                        cls_mask = cv2.resize(cls_mask, reshape_size)
                    cls_masks.append(cls_mask)
    return np.array(imgs), np.array(det_masks), np.array(cls_masks)


def _image_normalization(image):
    img = image / 255.
    img -= np.mean(img, keepdims=True)
    img /= (np.std(img, keepdims=True) + 1e-7)
    return img


def check_directory(file_path):
    """
    make new file on path if file is not already exist.
    :param file_path: file_path can be list of files to create.
    """
    if _isArrayLike(file_path):
        for file in file_path:
            if not os.path.exists(file):
                os.makedirs(file)
    else:
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def check_cv2_imwrite(file_path, file):
    if not os.path.exists(file_path):
        cv2.imwrite(file_path, file)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  #res = []
  #for k in topk:
  correct_k = correct[:1].view(-1).float().sum(0)
  res = correct_k.mul_(100.0/batch_size)
  return res

def get_metrics(gt, pred, r=3):
    # calculate precise, recall and f1 score
    gt = np.array(gt).astype('int')
    if pred == []:
        if gt.shape[0] == 0:
            return 1, 1, 1
        else:
            return 0, 0, 0

    pred = np.array(pred).astype('int')

    temp = np.concatenate([gt, pred])

    if temp.shape[0] != 0:
        x_max = np.max(temp[:, 0]) + 1
        y_max = np.max(temp[:, 1]) + 1

        gt_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(gt.shape[0]):
            x = gt[i, 0]
            y = gt[i, 1]
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(x_max, x + r)
            y2 = min(y_max, y + r)
            gt_map[y1:y2, x1:x2] = 1

        pred_map = np.zeros((y_max, x_max), dtype='int')
        for i in range(pred.shape[0]):
            x = pred[i, 0]
            y = pred[i, 1]
            pred_map[y, x] = 1

        result_map = gt_map * pred_map
        tp = result_map.sum()

        precision = tp / (pred.shape[0])  # + epsilon)
        recall = tp / (gt.shape[0])  # + epsilon)
        f1_score = 2 * (precision * recall / (precision + recall))  # + epsilon))

        return precision, recall, f1_score


def mask_to_corrdinates(mask):
    a = np.where(mask == 1)
    x = []
    for i, num in enumerate(a[0]):
        c = (a[1][i], num)
        x.append(c)
    print(x)


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def make_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.learning_rate
        weight_decay = args.weight_decay
        #if "bias" in key:
        #    lr = args.lr * args.weight_decay
        #    weight_decay = args.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)
    return optimizer

def _data_transforms_cell(args):

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.
    transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    return train_transform, valid_transform


if __name__ == '__main__':
    import keras.backend as K

    print(K.get_session().list_devices())
"""
    p = '/home/yichen/Desktop/sfcn-opi-yichen/data'
    from PIL import Image, ImageEnhance
    load_mask = Image.open('/home/yichen/Desktop/sfcn-opi-yichen/data/cls_and_det/train/img1/img1_detection.bmp')
    contrast = ImageEnhance.Contrast(load_mask)
    contrast.enhance(200).show(load_mask)
    maskt = cv2.imread('/home/yichen/Desktop/sfcn-opi-yichen/data/cls_and_det/train/img1/img1_detection.bmp', 0)
    mask_to_corrdinates(maskt)
    mat = loadmat('/home/yichen/Desktop/sfcn-opi-yichen/data/cls_and_det/train/img1/img1_detection.mat')['detection']
    print(np.sort(mat, axis=-1))

    p = '/home/yichen/Desktop/sfcn-opi-yichen/data'
    print(os.path.abspath('data'))
    #_reorder_image_files('/home/yichen/Desktop/sfcn-opi-yichen/data/cls_and_det')
    train_test_split(p, notation_type='point')
    from PIL import Image, ImageEnhance
    i = 1"""