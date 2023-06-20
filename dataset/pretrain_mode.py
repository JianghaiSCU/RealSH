import torch
import torch.nn as nn
from DLT import DLT_solve
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def subspace_project(input, vectors):
    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1)
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1))
    output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)

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
        self.relu = nn.LeakyReLU(inplace=False)
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

    def forward(self, image_patches, h4pt):

        img1_patch, img2_patch = image_patches[:, :1, :, :], image_patches[:, 1:, :, :]

        batch_size, _, h_patch, w_patch = img1_patch.size()

        img1_patch_fea, img2_patch_fea = self.share_feature(img1_patch), self.share_feature(img2_patch)

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

        return Homo_b, Homo_f


def fetch_net(params):
    if params.net_type == "BasesHomo":
        HNet = Net(params=params)
    else:
        raise NotImplementedError
    return HNet


def load_model(params):
    device = torch.device("cuda:{}".format(torch.cuda.device_count() - 1))

    model = fetch_net(params)
    state = torch.load(params.checkpoints_path, map_location='cpu')

    if "state_dict" in state and model is not None:
        try:
            model.load_state_dict(state["state_dict"])
            print("ckpt load")

        except:
            print("Using custom loading net")
            net_dict = model.state_dict()

            if "module" not in list(state["state_dict"].keys())[0]:
                state_dict = {"module." + k: v for k, v in state["state_dict"].items() if
                              "module." + k in net_dict.keys()}
            else:
                state_dict = {k.replace("module.", ""): v for k, v in state["state_dict"].items() if
                              k.replace("module.", "") in net_dict.keys()}

            net_dict.update(state_dict)
            model.load_state_dict(net_dict, strict=False)

    model.eval().to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model
