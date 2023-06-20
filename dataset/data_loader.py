import logging
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data
from .pre_processing import syn_data
from .pretrain_mode import load_model

_logger = logging.getLogger(__name__)

def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class UnHomoTrainData(Dataset):

    def __init__(self, params, phase='train'):
        assert phase in ['train', 'val', 'test']
        # 参数预设

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.params = params
        self.crop_size = self.params.crop_size

        self.rho = self.params.rho
        self.normalize = True

        self.image_list = os.path.join(self.params.data_dir, 'train.txt')
        self.image_path = os.path.join(self.params.data_dir, "img")
        self.mask_path = os.path.join(self.params.data_dir, "mask")

        self.data_infor = open(self.image_list, 'r').readlines()

        self.seed = 0
        random.seed(self.seed)
        random.shuffle(self.data_infor)

        self.model = load_model(self.params)
        # others

    def __len__(self):
        return len(self.data_infor)

    def __getitem__(self, idx):
        # img loading

        img_names = self.data_infor[idx].replace('\n', '')
        img_names = img_names.split(' ')

        img1 = cv2.imread(os.path.join(self.image_path, img_names[0]))
        img2 = cv2.imread(os.path.join(self.image_path, img_names[1]))

        mask_name = img_names[0].split('.')[0] + '_' + img_names[1].split('.')[0] + '.npy'
        mask = np.load(os.path.join(self.mask_path, mask_name), allow_pickle=True)

        img2_new, img2_random, homo_gt, homo_inv = syn_data(img1, img2, self.model, mask, self.params)

        img1, img2_new, img2, img2_random, img1_patch, img2_new_patch, img2_patch, img2_random_patch, \
        h4pt, offset_gt_b, offset_gt_f, start = \
            self.data_aug(img1, img2_new, img2, img2_random, homo_gt)

        imgs_gray_full = torch.cat((img1, img2_new, img2, img2_random), dim=2).permute(2, 0, 1).float()
        imgs_gray_patch = torch.cat((img1_patch, img2_new_patch, img2_patch, img2_random_patch),
                                    dim=2).permute(2, 0, 1).float()
        offset_gt = torch.cat((offset_gt_b, offset_gt_f), dim=0)
        start = torch.Tensor(start).reshape(2, 1, 1).float()
        homo_inv = torch.Tensor(homo_inv).float()
        # output dict
        data_dict = {"imgs_gray_full": imgs_gray_full, "imgs_gray_patch": imgs_gray_patch,
                     "h4pt": h4pt, "start": start, "offset_gt": offset_gt, "homo_inv": homo_inv}

        return data_dict

    def data_aug(self, img1, img2_new, img2, img2_random, homo_gt, start=None, normalize=True, gray=True):

        def random_crop_tt(img1, img2_new, img2, img2_random, homo_gt, start):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size

            if start is None:
                x = random.randint(self.rho, width - self.rho - patch_size_w)
                y = random.randint(self.rho, height - self.rho - patch_size_h)
                start = [x, y]
            else:
                x, y = start
            img1_patch = img1[y: y + patch_size_h, x: x + patch_size_w, :]
            img2_new_patch = img2_new[y: y + patch_size_h, x: x + patch_size_w, :]
            img2_random_patch = img2_random[y: y + patch_size_h, x: x + patch_size_w, :]
            img2_patch = img2[y: y + patch_size_h, x: x + patch_size_w, :]

            top_left_point = (x, y)
            bottom_left_point = (x, y + patch_size_h - 1)
            bottom_right_point = (patch_size_w + x - 1, patch_size_h + y - 1)
            top_right_point = (x + patch_size_w - 1, y)
            h4pt = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
            h4pt_ = np.reshape(h4pt.copy(), (1, 4, 2)).astype('float32')

            offset_gt_b = cv2.perspectiveTransform(h4pt_, homo_gt) - h4pt_
            offset_gt_f = cv2.perspectiveTransform(h4pt_, np.linalg.inv(homo_gt)) - h4pt_

            h4pt = torch.tensor(np.reshape(h4pt, (-1))).float()
            offset_gt_b = torch.tensor(np.reshape(offset_gt_b, (1, -1))).float()
            offset_gt_f = torch.tensor(np.reshape(offset_gt_f, (1, -1))).float()

            return img1, img2_new, img2, img2_random, img1_patch, img2_new_patch, img2_patch,\
                   img2_random_patch, h4pt, offset_gt_b, offset_gt_f, start

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2_new = (img2_new - self.mean_I) / self.std_I
            img2_random = (img2_random - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2_new = np.mean(img2_new, axis=2, keepdims=True)
            img2_random = np.mean(img2_random, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1, img2_new, img2, img2_random = list(map(torch.Tensor, [img1, img2_new, img2, img2_random]))

        img1, img2_new, img2, img2_random, img1_patch, img2_new_patch, img2_patch, \
        img2_random_patch, h4pt, offset_gt_b, offset_gt_f, start = \
            random_crop_tt(img1, img2_new, img2, img2_random, homo_gt, start)

        return img1, img2_new, img2, img2_random, img1_patch, img2_new_patch, img2_patch, \
               img2_random_patch, h4pt, offset_gt_b, offset_gt_f, start


class HomoTestData(Dataset):
    def __init__(self, params, phase):
        assert phase in ["test", "val"]

        self.params = params

        self.patch_size_h, self.patch_size_w = self.params.crop_size

        self.data_list = os.path.join(self.params.data_dir, "test.txt")
        self.npy_path = os.path.join(self.params.data_dir, "npy")
        self.image_path = os.path.join(self.params.data_dir, "img")

        self.data_infor = open(self.data_list, 'r').readlines()

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')

        video_names = img_names.split('/')[0]
        img_names = img_names.split(' ')

        pt_names = img_names[0].split('/')[-1] + '_' + img_names[1].split('/')[-1] + '.npy'

        img1 = cv2.imread(os.path.join(self.image_path, img_names[0]))
        img2 = cv2.imread(os.path.join(self.image_path, img_names[1]))

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)), dim=-1).permute(2, 0, 1).float()
        height, width, _ = img1.shape

        pt_set = np.load(os.path.join(self.npy_path, pt_names), allow_pickle=True)
        pt_set = str(pt_set.item())

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)  # 变均值，灰度
        img2 = np.mean(img2, axis=2, keepdims=True)

        x = 32  # patch should in the middle of full img when testing
        y = 20  # patch should in the middle of full img when testing
        start = torch.Tensor([x, y]).reshape(2, 1, 1).float()

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_size_h - 1)
        bottom_right_point = (self.patch_size_w + x - 1, self.patch_size_h + y - 1)
        top_right_point = (x + self.patch_size_w - 1, y)
        h4pt = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        h4pt = np.reshape(h4pt, (-1))
        h4pt = torch.tensor(h4pt).float()

        img1, img2 = list(map(torch.Tensor, [img1, img2]))
        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0, 1).float()
        imgs_gray_patch = imgs_gray_full[:, y: y + self.patch_size_h, x: x + self.patch_size_w]

        data_dict = {"imgs_gray_full": imgs_gray_full, "imgs_full": imgs_full,
                     "imgs_gray_patch": imgs_gray_patch, "h4pt": h4pt, "pt_set": pt_set,
                     "video_names": video_names, "start": start}
        return data_dict


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    train_ds = UnHomoTrainData(params, phase='train')
    val_ds = HomoTestData(params, phase='val')
    test_ds = HomoTestData(params, phase='test')

    dataloaders = {}
    # add defalt train data loader
    train_dl = DataLoader(
        train_ds,
        batch_size=params.train_batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda,
        drop_last=True,
        worker_init_fn=worker_init_fn)
    dataloaders["train"] = train_dl

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                dl = DataLoader(
                    val_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda
                )
            elif split == "test":
                dl = DataLoader(
                    test_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    pin_memory=params.cuda
                )
            else:
                raise ValueError("Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
