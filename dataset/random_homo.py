import torch
import numpy as np
import cv2
import os
import torch
from dataset.homo_sampling import Synthetic_Homo, Synthetic_HomoFlow


def get_grid(batch_size, H, W, start=0):
    if type(start) is not int:
        xx = torch.arange(0, W).to(start.device)
        yy = torch.arange(0, H).to(start.device)
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    # ones = torch.ones_like(xx).to(start.device) if type(start) is not int else torch.ones_like(xx)
    grid = torch.cat((xx, yy), 1).float()

    grid[:, :2, :, :] = grid[:, :2, :, :] + start

    return grid


def multi_corner_DLT(src_p, off_set):
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)

    bs, _ = src_p.shape[:2]
    divide = int(np.sqrt(len(src_p[0]) / 2) - 1)
    row_num = (divide + 1) * 2
    src_ps = src_p
    off_sets = off_set
    for i in range(divide):
        for j in range(divide):
            h4p = src_p[:, [2 * j + row_num * i, 2 * j + row_num * i + 1,
                            2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1,
                            2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                            2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4, 2)

            pred_h4p = off_set[:, [2 * j + row_num * i, 2 * j + row_num * i + 1,
                                   2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1,
                                   2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                                   2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4,
                                                                                                              2)

            if i + j == 0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis=1)
                off_sets = torch.cat((off_sets, pred_h4p), axis=1)

    bs, n, h, w = src_ps.shape

    N = bs * n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, h, 1)
    if off_set.is_cuda:
        ones = ones.to(off_set.device)
    xy1 = torch.cat((src_ps, ones), 2)
    zeros = torch.zeros_like(xy1)
    if off_set.is_cuda:
        zeros = zeros.to(off_set.device)

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.linalg.pinv(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)

    return H


def flow2offset(flow):
    b, c, h, w = flow.shape
    x, y = 0, 0

    flow_ = flow.detach().permute(0, 2, 3, 1).squeeze(0)

    offset_top_left_point = flow_[y, x].unsqueeze(0)
    offset_bottom_left_point = flow_[y + h - 1, x].unsqueeze(0)
    offset_bottom_right_point = flow_[h + y - 1, w + x - 1].unsqueeze(0)
    offset_top_right_point = flow_[y, x + w - 1].unsqueeze(0)
    offset = torch.cat((offset_top_left_point, offset_bottom_left_point,
                        offset_bottom_right_point, offset_top_right_point)).unsqueeze(0)

    return offset


def flow_gen(image, params):
    h, w, c = image.shape

    homo_sampling_module = Synthetic_Homo(img_size=(h, w), random_hom=params.random_homo)
    synthetic_flow_generator = Synthetic_HomoFlow(img_size=(h, w), sampling_module=homo_sampling_module)
    flow_gt = synthetic_flow_generator() # .cuda()
    flow_gt = flow_gt.requires_grad_(False)

    return flow_gt


def random_homo_gen(image, params):
    h, w, c = image.shape

    flow = flow_gen(image, params)
    grid = get_grid(1, h, w)
    original = flow2offset(grid).reshape(1, -1).type(torch.float64)
    offset = flow2offset(flow).reshape(1, -1)

    homo = multi_corner_DLT(original, offset).squeeze(0).squeeze(0).detach().cpu().numpy()

    return homo
