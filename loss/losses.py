import numpy as np
import torch
import torch.nn as nn
import cv2
import kornia.geometry as ge


class LossL1(nn.Module):
    def __init__(self, reduction='mean'):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


def non_zero_selection(score, x, y):

    output = []
    reference = []
    for i in range(score.nonzero().shape[0]):
        x_ = x[score.nonzero()[i][0]].unsqueeze(0)
        y_ = y[score.nonzero()[i][0]].unsqueeze(0)
        output.append(x_)
        reference.append(y_)

    output = torch.cat(output, dim=0)
    reference = torch.cat(reference, dim=0)

    return output, reference


def compute_losses(data, output, params):
    loss = {}

    offset_gt_b, offset_gt_f = data["offset_gt"][:, 0, :], \
                               data["offset_gt"][:, 1, :]
    homo_inv = data["homo_inv"]

    img2_patch_recon, img2_patch_recon_fea = output["img2_patch_recon"], \
                                             output["img2_patch_recon_fea"]
    img2_patch_ori, img2_patch_ori_fea = data["imgs_gray_patch"][:, 2:3, :, :], \
                                         output["img2_patch_ori_fea"]

    offset_pred_b, offset_pred_f = output["offset_b"], \
                                   output["offset_f"]

    img2_patch_new_score, img2_patch_ori_score, img2_patch_random_score = \
        output["img2_patch_new_score"].ge(0.5).float(), \
        output["img2_patch_ori_score"], \
        output["img2_patch_random_score"]

    ones_score = torch.ones_like(img2_patch_new_score.detach()).to(img2_patch_new_score.device)
    zeros_score = torch.zeros_like(img2_patch_new_score.detach()).to(img2_patch_new_score.device)

    b, _, h_patch, w_patch = img2_patch_ori.shape
    # Loss Definition
    sup_criterion = LossL1(reduction='mean')
    select_criterion = nn.BCELoss()

    warp_img2_patch_recon_fea = ge.warp_perspective(img2_patch_recon_fea, homo_inv, dsize=(h_patch, w_patch))

    loss["photo_loss"] = params.inpainting_weights * sup_criterion(warp_img2_patch_recon_fea, img2_patch_ori_fea)

    loss['score'] = params.selection_weights * (select_criterion(img2_patch_ori_score, ones_score) +
                                                select_criterion(img2_patch_random_score, zeros_score))

    if img2_patch_new_score.nonzero().shape[0] == 0:

        loss['supervise'] = sup_criterion(offset_pred_f, offset_gt_f) + \
                            sup_criterion(offset_pred_b, offset_gt_b)

    else:
        offset_pred_f_, offset_gt_f_ = non_zero_selection(img2_patch_new_score, offset_pred_f, offset_gt_f)
        offset_pred_b_, offset_gt_b_ = non_zero_selection(img2_patch_new_score, offset_pred_b, offset_gt_b)

        loss['supervise'] = sup_criterion(offset_pred_f_, offset_gt_f_) + \
                            sup_criterion(offset_pred_b_, offset_gt_b_)

    loss['total'] = loss['supervise'] + loss['photo_loss'] + loss['score']

    return loss


def ComputeErrH_v2(src, dst, H):
    '''
    :param src: B, N, 2
    :param dst: B, N, 2
    :param H: B, 3, 3
    '''
    src, dst = src.unsqueeze(0).unsqueeze(0), \
               dst.unsqueeze(0).unsqueeze(0)
    src_warp = ge.transform_points(H.unsqueeze(0), src)

    err = torch.linalg.norm(dst-src_warp)
    return err


def compute_eval_results(data_batch, output_batch):

    imgs_full = data_batch["imgs_gray_full"]

    pt_set = list(map(eval, data_batch["pt_set"]))
    pt_set = list(map(lambda x: x['matche_pts'], pt_set))

    batch_size, _, img_h, img_w = imgs_full.shape
    Homo_b = output_batch["Homo_b"]
    Homo_f = output_batch["Homo_f"]

    errs_m = []

    for i in range(batch_size):
        pts = torch.Tensor(pt_set[i]).to(imgs_full.device)
        err = 0
        for j in range(6):
            src, dst = pts[j][0], pts[j][1]
            pred_err = min(ComputeErrH_v2(src=src, dst=dst, H=Homo_b[i]),
                           ComputeErrH_v2(src=dst, dst=src, H=Homo_f[i]))
            err += pred_err
        err /= 6
        errs_m.append(err)

    eval_results = {"errors_m": errs_m}

    return eval_results

