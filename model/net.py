import torch
import torch.nn as nn

from DLT import DLT_solve
from .utils import get_warp_flow, upsample2d_flow_as, get_grid
from .module.aspp import ASPP
import torch.nn.functional as F
import warnings
import kornia.geometry as ge

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ['Net']


def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def tensor_dilation(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    dilation, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilation


def gen_basis(h, w, is_qr=True, is_scale=True):
    basis_nb = 8
    grid = get_grid(1, h, w).permute(0, 2, 3, 1).contiguous()  # 1, w, h, (x, y, 1)
    flow = grid[:, :, :, :2] * 0

    names = globals()
    for i in range(1, basis_nb + 1):
        names['basis_' + str(i)] = flow.clone()

    basis_1[:, :, :, 0] += grid[:, :, :, 0]  # [1, w, h, (x, 0)]
    basis_2[:, :, :, 0] += grid[:, :, :, 1]  # [1, w, h, (y, 0)]
    basis_3[:, :, :, 0] += 1  # [1, w, h, (1, 0)]
    basis_4[:, :, :, 1] += grid[:, :, :, 0]  # [1, w, h, (0, x)]
    basis_5[:, :, :, 1] += grid[:, :, :, 1]  # [1, w, h, (0, y)]
    basis_6[:, :, :, 1] += 1  # [1, w, h, (0, 1)]
    basis_7[:, :, :, 0] += grid[:, :, :, 0] ** 2  # [1, w, h, (x^2, xy)]
    basis_7[:, :, :, 1] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (x^2, xy)]
    basis_8[:, :, :, 0] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (xy, y^2)]
    basis_8[:, :, :, 1] += grid[:, :, :, 1] ** 2  # [1, w, h, (xy, y^2)]

    flows = torch.cat([names['basis_' + str(i)] for i in range(1, basis_nb + 1)], dim=0)
    if is_qr:
        flows_ = flows.view(basis_nb, -1).permute(1, 0).contiguous()  # N, h, w, c --> N, h*w*c --> h*w*c, N
        flow_q, _ = torch.qr(flows_)
        flow_q = flow_q.permute(1, 0).reshape(basis_nb, h, w, 2).contiguous()
        flows = flow_q

    if is_scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    return flows.permute(0, 3, 1, 2).contiguous()


def subspace_project(input, vectors):
    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1).contiguous()
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1)).contiguous()
    output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_).contiguous()

    return output


class Subspace(nn.Module):

    def __init__(self, ch_in, k=16, use_SVD=True, use_PCA=False):
        super(Subspace, self).__init__()
        self.k = k
        self.Block = SubspaceBlock(ch_in, self.k)
        self.use_SVD = use_SVD
        self.use_PCA = use_PCA

    def forward(self, x):
        sub = self.Block(x)
        x = subspace_project(x, sub)

        return x


class SubspaceBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(SubspaceBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=False)

        self.conv0 = conv(inplanes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False)
        self.bn0 = nn.BatchNorm2d(planes)
        self.conv1 = conv(planes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.relu(self.bn0(self.conv0(x)))

        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ShareFeature(nn.Module):

    def __init__(self, num_chs):
        super(ShareFeature, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(num_chs, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),

            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),

            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(Discriminator, self).__init__()
        self.cls_head = self.cls_net(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def cls_net(input_channels, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels * 2, 32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, stride=2, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cls_head(x)
        bs = len(x)
        x = self.conv_last(x)
        x = self.pool(x).view(bs, -1)
        return x


def training_warp(image, homo, start, patch_size):
    b, _, h, w = image.shape
    patch_size_h, patch_size_w = patch_size

    warped_image = []
    warped_img = ge.warp_perspective(image, homo.float(), dsize=(h, w))

    for b_ in range(b):
        x, y = start[b_].squeeze(-1).squeeze(-1).detach().cpu().numpy()
        x, y = int(x), int(y)
        img_ = warped_img[b_].unsqueeze(0)
        img_patch_ = img_[:, :, y: y + patch_size_h, x: x + patch_size_w]

        warped_image.append(img_patch_)

    warped_images = torch.cat(warped_image, dim=0)

    return warped_images


class Net(nn.Module):
    # 224*224
    def __init__(self, params):
        super(Net, self).__init__()

        self.params = params

        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.basis_vector_num = 16
        self.crop_size = self.params.crop_size

        self.share_feature = ShareFeature(1)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.block = BasicBlock
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.sp_layer3 = Subspace(256)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.sp_layer4 = Subspace(512)
        self.subspace_block = SubspaceBlock(2, self.basis_vector_num)

        self.recon = self.reconstruction(32)
        self.select = self.selection(16)
        self.sigmoid = nn.Sigmoid()

        self.conv_last = nn.Conv2d(512, 8, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def reconstruction(input_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=input_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def selection(input_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels=1, out_channels=input_channels,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=input_channels,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=1,
                                kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def forward(self, data_batch):

        h4pt = data_batch['h4pt']

        if self.training:
            img1_patch, img2_patch, img2_patch_ori, img2_patch_random = \
                data_batch['imgs_gray_patch'][:, :1, ...], \
                data_batch['imgs_gray_patch'][:, 1:2, ...], \
                data_batch['imgs_gray_patch'][:, 2:3, ...], \
                data_batch['imgs_gray_patch'][:, 3:4, ...]

            img2_patch = self.recon(img2_patch)
            img1_patch_fea, img2_patch_fea, img2_patch_ori_fea = self.share_feature(img1_patch), \
                                                                 self.share_feature(img2_patch), \
                                                                 self.share_feature(img2_patch_ori)
            batch_size, _, h_patch, w_patch = img1_patch.size()
            img2_patch_new_score, img2_patch_ori_score, img2_patch_random_score = \
                self.sigmoid(self.select(img2_patch).view(batch_size, -1)), \
                self.sigmoid(self.select(img2_patch_ori).view(batch_size, -1)), \
                self.sigmoid(self.select(img2_patch_random).view(batch_size, -1))
        else:
            img1_patch, img2_patch = data_batch['imgs_gray_patch'][:, :1, ...], \
                                     data_batch['imgs_gray_patch'][:, 1:2, ...]
            img1_patch_fea, img2_patch_fea, img2_patch_ori_fea = self.share_feature(img1_patch), \
                                                                 self.share_feature(img2_patch), \
                                                                 None
            img2_patch_new_score, img2_patch_ori_score, img2_patch_random_score = None, None, None
            batch_size, _, h_patch, w_patch = img1_patch.size()

        x = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sp_layer3(self.layer3(x))
        x = self.sp_layer4(self.layer4(x))
        x = self.conv_last(x)  # bs,8,h,w
        offset_f = self.pool(x).view(batch_size, -1)  # bs,8,1,1
        Homo_f = DLT_solve(h4pt, offset_f).squeeze(1)

        x = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sp_layer3(self.layer3(x))
        x = self.sp_layer4(self.layer4(x))
        x = self.conv_last(x)  # bs,8,h,w
        offset_b = self.pool(x).view(batch_size, -1)  # bs,8,1,1
        Homo_b = DLT_solve(h4pt, offset_b).squeeze(1)

        return {'img2_patch_recon': img2_patch,
                'img2_patch_recon_fea': img2_patch_fea,
                'img2_patch_ori_fea': img2_patch_ori_fea,
                'Homo_b': Homo_b, 'Homo_f': Homo_f,
                'offset_f': offset_f, 'offset_b': offset_b,
                'img2_patch_new_score': img2_patch_new_score,
                'img2_patch_ori_score': img2_patch_ori_score,
                'img2_patch_random_score': img2_patch_random_score}


def fetch_net(params):
    if params.net_type == "BasesHomo":
        model = Net(params)
    else:
        raise NotImplementedError
    return model
