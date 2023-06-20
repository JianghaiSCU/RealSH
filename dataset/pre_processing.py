import torch
import cv2
import numpy as np
import random
from .random_homo import random_homo_gen


def get_input_pair(img1, img2, params):

    patch_size_h, patch_size_w = params.crop_size[0], params.crop_size[1]

    height, width, _ = img1.shape

    mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
    std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

    img1 = (img1 - mean_I) / std_I
    img2 = (img2 - mean_I) / std_I

    img1 = np.mean(img1, axis=2, keepdims=True)  # 变均值，灰度
    img2 = np.mean(img2, axis=2, keepdims=True)

    x = 32
    y = 20

    top_left_point = (x, y)
    bottom_left_point = (x, y + patch_size_h - 1)
    bottom_right_point = (patch_size_w + x - 1, patch_size_h + y - 1)
    top_right_point = (x + patch_size_w - 1, y)
    h4pt = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    h4pt = np.reshape(h4pt, (-1))
    h4pt = torch.tensor(h4pt).unsqueeze(0).float()

    # if img1.shape[0] != self.crop_size[0] or img1.shape[1] != self.crop_size[1]:
    img1, img2 = list(map(torch.Tensor, [img1, img2]))
    imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0, 1).unsqueeze(0).float()
    imgs_gray_patch = imgs_gray_full[:, :, y: y + patch_size_h, x: x + patch_size_w]

    return imgs_gray_patch, h4pt


def syn_data(img1, img2, model, mask, params):
    device = torch.device("cuda:{}".format(torch.cuda.device_count() - 1))

    h, w, c = img1.shape

    imgs_patch, h4pt = get_input_pair(img1, img2, params)

    with torch.no_grad():

        Homo_b, Homo_f = model(imgs_patch.to(device), h4pt.to(device))

        Homo_b, Homo_f = Homo_b.squeeze(0).detach().cpu().numpy(), \
                         Homo_f.squeeze(0).detach().cpu().numpy()

        forward_mask = (1 - mask).astype('float32')
        backward_mask = mask.astype('float32')

        Homo_gt = random_homo_gen(img1, params)

        Homo_gt_inv = np.linalg.inv(Homo_gt)
        Homo_inv = np.matmul(Homo_b, Homo_gt_inv)

        source_image_warp = cv2.warpPerspective(img1, Homo_gt, (w, h))

        Homo_t = np.matmul(Homo_gt, Homo_f)
        Homo_random = random_homo_gen(img1, params)
        Homo_random = np.matmul(Homo_random, Homo_t)

        target_image_warp = cv2.warpPerspective(img2, Homo_t, (w, h))
        target_image_random = cv2.warpPerspective(img2, Homo_random, (w, h))

        forward_mask_warp = np.expand_dims(cv2.warpPerspective(forward_mask, Homo_gt, (w, h)), axis=-1)
        backward_mask_warp = np.expand_dims(cv2.warpPerspective(backward_mask, Homo_gt, (w, h)), axis=-1)

        target_image_new = (source_image_warp * backward_mask_warp +
                            target_image_warp * forward_mask_warp).astype(img1.dtype)

        target_image_random = (source_image_warp * backward_mask_warp +
                               target_image_random * forward_mask_warp).astype(img1.dtype)

        return target_image_new, target_image_random, Homo_gt, Homo_inv
