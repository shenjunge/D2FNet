import torch
import torch.nn as nn
from vit_model import VisionTransformer
import torch.nn.functional as F
import cv2 as cv
import numpy as  np


class spatial_attention(nn.Module):
    def __init__(self, out_channal=1):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(2, out_channal, (3, 3), padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max, _ = torch.max(x, 1, True)
        x_mean = torch.mean(x, 1, True)
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class channal_attention(nn.Module):
    def __init__(self, in_channal, ratio=4):
        super(channal_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channal, in_channal//ratio, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channal//ratio, in_channal, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        x_avg = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        x = x_avg + x_max
        return self.sigmoid(x)


class CBAM_block(nn.Module):
    def __init__(self, in_channal=64, out_channal=1):
        super(CBAM_block, self).__init__()
        self.spatial_block = spatial_attention(out_channal)
        self.channal_block = channal_attention(in_channal)

    def forward(self, x):
        x_cam = self.channal_block(x)
        x_channal_attention = x_cam * x
        x_sam = self.spatial_block(x_channal_attention)
        x_out = x_sam * x_channal_attention
        return x_out


class CNN_block(nn.Module):
    def __init__(self, in_channal, out_channal):
        super(CNN_block, self).__init__()
        self.in_channal = in_channal
        self.out_channal = out_channal
        self.conv = nn.Conv2d(self.in_channal, self.out_channal, (3, 3), (1, 1), 1)
        self.bn = nn.BatchNorm2d(self.out_channal)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class fra_sig_network(nn.Module):
    def __init__(self, vit_weight='', mode='train', num_out=7):
        super(fra_sig_network, self).__init__()
        self.mode = mode
        self.block1 = CNN_block(1, 16)
        self.block2 = CNN_block(16, 32)
        self.block3 = CNN_block(32, 64)
        self.jump_cnn = nn.Conv2d(1, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)
        self.cabm = CBAM_block(64)

        self.vit = VisionTransformer(depth=3)
        # 加载参数
        '''save_model = torch.load(vit_weight)
        model_dict = self.vit.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.vit.load_state_dict(model_dict)
        # 冻结参数
        for name, parameter in self.vit.named_parameters():
            parameter.requires_grad = False'''

        self.fusion_conv = nn.Conv2d(67, 1, (1, 1), padding=0)
        self.fusion_relu = nn.ReLU()
        self.norm_2d = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(4096, 512)
        self.liner_norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.liner_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_out)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x_sig, x_fra):
        jump_x = F.relu(self.jump_norm(self.jump_cnn(x_sig)))
        x_sig = self.block1(x_sig)
        x_sig = self.block2(x_sig)
        x_sig = self.block3(x_sig)
        x_sig = x_sig + jump_x
        x_sig = self.cabm(x_sig)

        x_fra = self.vit(x_fra)
        x_fra = x_fra.reshape(x_fra.shape[0], 3, 224, 224)
        x_fra = F.interpolate(x_fra, size=[64, 64])
        x = x_fra[0].permute(1, 2, 0).cpu().detach().numpy()
        x_max, x_min = np.max(x), np.min(x)
        x = np.int64(((x-x_min)/(x_max-x_min))*255)
        cv.imwrite('./att_map.png', x)
        cv.imshow('a', x)
        cv.waitKey()
        fusion_x = torch.cat([x_fra, x_sig], dim=1)

        fusion_x = self.fusion_relu(self.norm_2d(self.fusion_conv(fusion_x)))
        flatten_x = fusion_x.view(fusion_x.size(0), -1)
        x = self.fusion_relu(self.liner_norm1(self.fc1(flatten_x)))
        if self.mode == 'train':
            x = self.drop_out(x)
        x = self.fusion_relu(self.liner_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x, flatten_x


class time_fra(nn.Module):
    def __init__(self, mode='train', num_out=7):
        super(time_fra, self).__init__()
        self.mode = mode
        self.block1 = CNN_block(1, 16)
        self.block2 = CNN_block(16, 32)
        self.block3 = CNN_block(32, 64)
        self.jump_cnn = nn.Conv2d(1, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)

        self.block4 = CNN_block(3, 16)
        self.block5 = CNN_block(16, 32)
        self.block6 = CNN_block(32, 64)
        self.jump_cnn_fra = nn.Conv2d(3, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)

        self.fusion_conv = nn.Conv2d(128, 1, (1, 1), padding=0)
        self.fusion_relu = nn.ReLU()
        self.norm_2d = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(4096, 512)
        self.liner_norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.liner_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_out)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x_sig, x_fra):
        jump_x = F.relu(self.jump_norm(self.jump_cnn(x_sig)))
        x_sig = self.block1(x_sig)
        x_sig = self.block2(x_sig)
        x_sig = self.block3(x_sig)
        x_sig = x_sig + jump_x

        jump_x_fra = F.relu(self.jump_norm(self.jump_cnn_fra(x_fra)))
        x_fra = self.block4(x_fra)
        x_fra = self.block5(x_fra)
        x_fra = self.block6(x_fra)
        x_fra = x_fra + jump_x_fra
        x_fra = F.interpolate(x_fra, size=[64, 64])
        fusion_x = torch.cat([x_fra, x_sig], dim=1)

        fusion_x = self.fusion_relu(self.norm_2d(self.fusion_conv(fusion_x)))
        flatten_x = fusion_x.view(fusion_x.size(0), -1)
        x = self.fusion_relu(self.liner_norm1(self.fc1(flatten_x)))
        if self.mode == 'train':
            x = self.drop_out(x)
        x = self.fusion_relu(self.liner_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


class time_model(nn.Module):
    def __init__(self, mode='train', num_out=7):
        super(time_model, self).__init__()
        self.mode = mode
        self.block1 = CNN_block(1, 16)
        self.block2 = CNN_block(16, 32)
        self.block3 = CNN_block(32, 64)
        self.jump_cnn = nn.Conv2d(1, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)
        self.cabm = CBAM_block(64, 1)

        self.fusion_conv = nn.Conv2d(64, 1, (1, 1), padding=0)
        self.fusion_relu = nn.ReLU()
        self.norm_2d = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(4096, 512)
        self.liner_norm1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.liner_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_out)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x_sig):
        jump_x = F.relu(self.jump_norm(self.jump_cnn(x_sig)))
        x_sig = self.block1(x_sig)
        x_sig = self.block2(x_sig)
        x_sig = self.block3(x_sig)
        x_sig = x_sig + jump_x
        x_sig = self.cabm(x_sig)

        x_sig = self.fusion_relu(self.norm_2d(self.fusion_conv(x_sig)))

        flatten_x = x_sig.view(x_sig.size(0), -1)
        x = self.relu(self.liner_norm1(self.fc1(flatten_x)))
        if self.mode == 'train':
            x = self.drop_out(x)
        x = self.relu(self.liner_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


class fra_model(nn.Module):
    def __init__(self, mode='train', num_out=7):
        super(fra_model, self).__init__()
        self.mode = mode

        self.vit = VisionTransformer(depth=3)

        self.fusion_conv = nn.Conv2d(3, 1, (1, 1), padding=0)
        self.fusion_relu = nn.ReLU()
        self.norm_2d = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(4096, 512)
        self.liner_norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.liner_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_out)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x_fra):
        x_fra = self.vit(x_fra)
        x_fra = x_fra.reshape(x_fra.shape[0], 3, 224, 224)
        x_fra = F.interpolate(x_fra, size=[64, 64])
        # fusion_x = torch.cat([x_fra, x_sig], dim=1)

        fusion_x = self.fusion_relu(self.norm_2d(self.fusion_conv(x_fra)))
        flatten_x = fusion_x.view(fusion_x.size(0), -1)
        x = self.fusion_relu(self.liner_norm1(self.fc1(flatten_x)))
        if self.mode == 'train':
            x = self.drop_out(x)
        x = self.fusion_relu(self.liner_norm2(self.fc2(x)))
        x = self.fc3(x)

        return x


class orignal_network(nn.Module):
    def __init__(self, mode='train', num_out=7):
        super(orignal_network, self).__init__()
        self.mode = mode
        # time
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.jump_block = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # fra
        self.block4 = CNN_block(3, 16)
        self.block5 = CNN_block(16, 32)
        self.block6 = CNN_block(32, 64)
        self.jump_cnn_fra = nn.Conv2d(3, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)

        # fusion
        self.conv = nn.Conv1d(64, 1, 1, 1, 0)
        self.pool = nn.MaxPool1d(16)

        self.fc1 = nn.Linear(256, 128)
        self.liner_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.liner_norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_out)
        self.drop_out = nn.Dropout(0.5)


    def forward(self, x_sig, x_fra):
        x_jump_sig = self.jump_block(x_sig)
        x_sig = self.block1(x_sig)
        x_sig = self.block2(x_sig)
        x_sig = self.block3(x_sig)
        x_sig = x_sig + x_jump_sig

        x_jump_fra = F.relu(self.jump_norm(self.jump_cnn_fra(x_fra)))
        x_fra = self.block4(x_fra)
        x_fra = self.block5(x_fra)
        x_fra = self.block6(x_fra)
        x_fra = x_fra + x_jump_fra
        x_fra = F.interpolate(x_fra, size=[64, 64])
        x_fra = x_fra.view(x_fra.size(0), x_fra.size(1), -1)

        x_funsion = x_fra + x_sig
        x_funsion = self.pool(self.conv(x_funsion))
        x_funsion = x_funsion.view(x_funsion.size(0), -1)

        x = F.relu(self.liner_norm1(self.fc1(x_funsion)))
        if self.mode == 'train':
            x = self.drop_out(x)
        x = F.relu(self.liner_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


class time_fra_cbam(nn.Module):
    def __init__(self, mode='train', num_out=7):
        super(time_fra_cbam, self).__init__()
        self.mode = mode
        self.block1 = CNN_block(1, 16)
        self.block2 = CNN_block(16, 32)
        self.block3 = CNN_block(32, 64)
        self.jump_cnn = nn.Conv2d(1, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)
        self.cabm = CBAM_block(64)

        self.block4 = CNN_block(3, 16)
        self.block5 = CNN_block(16, 32)
        self.block6 = CNN_block(32, 64)
        self.jump_cnn_fra = nn.Conv2d(3, 64, (1, 1))
        self.jump_norm = nn.BatchNorm2d(64)
        self.cabm_fra = CBAM_block(64)

        self.fusion_conv = nn.Conv2d(128, 1, (1, 1), padding=0)
        self.fusion_relu = nn.ReLU()
        self.norm_2d = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(4096, 512)
        self.liner_norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.liner_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_out)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x_sig, x_fra):
        jump_x = F.relu(self.jump_norm(self.jump_cnn(x_sig)))
        x_sig = self.block1(x_sig)
        x_sig = self.block2(x_sig)
        x_sig = self.block3(x_sig)
        x_sig = x_sig + jump_x
        x_sig = self.cabm(x_sig)

        jump_x_fra = F.relu(self.jump_norm(self.jump_cnn_fra(x_fra)))
        x_fra = self.block4(x_fra)
        x_fra = self.block5(x_fra)
        x_fra = self.block6(x_fra)
        x_fra = x_fra + jump_x_fra
        x_fra = self.cabm_fra(x_fra)
        x_fra = F.interpolate(x_fra, size=[64, 64])
        fusion_x = torch.cat([x_fra, x_sig], dim=1)

        fusion_x = self.fusion_relu(self.norm_2d(self.fusion_conv(fusion_x)))
        flatten_x = fusion_x.view(fusion_x.size(0), -1)
        x = self.fusion_relu(self.liner_norm1(self.fc1(flatten_x)))
        if self.mode == 'train':
            x = self.drop_out(x)
        x = self.fusion_relu(self.liner_norm2(self.fc2(x)))
        x = self.fc3(x)
        return x


class fault_net(nn.Module):
    def __init__(self):
        super(fault_net, self).__init__()
        self.conv = nn.Conv2d(3, 32, (3, 3), (1, 1), padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # block1
        self.conv_b1 = nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1)
        self.bn_b1 = nn.BatchNorm2d(32)

        # block2
        self.conv_b2_1 = nn.Conv2d(32, 64, (3, 3), (2, 2), padding=1)
        self.bn_b2 = nn.BatchNorm2d(64)
        self.conv_b2_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)
        self.conv_b2_3 = nn.Conv2d(32, 64, (1, 1), (2, 2), padding=0)

        # block3
        self.conv_b3_1 = nn.Conv2d(64, 128, (3, 3), (2, 2), padding=1)
        self.bn_b3 = nn.BatchNorm2d(128)
        self.conv_b3_2 = nn.Conv2d(128, 128, (3, 3), (1, 1), padding=1)
        self.conv_b3_3 = nn.Conv2d(64, 128, (1, 1), (2, 2))

        # block4
        self.conv_b4_1 = nn.Conv2d(128, 256, (3, 3), (2, 2), padding=1)
        self.bn_b4 = nn.BatchNorm2d(256)
        self.conv_b4_2 = nn.Conv2d(256, 256, (3, 3), (1, 1), padding=1)
        self.conv_b4_3 = nn.Conv2d(128, 256, (1, 1), (2, 2))

        # block5
        self.conv_b5_1 = nn.Conv2d(256, 512, (3, 3), (2, 2), padding=1)
        self.bn_b5 = nn.BatchNorm2d(512)
        self.conv_b5_2 = nn.Conv2d(512, 512, (3, 3), (1, 1), padding=1)
        self.conv_b5_3 = nn.Conv2d(256, 512, (1, 1), (2, 2))

        # block6
        self.conv_b6_1 = nn.Conv2d(512, 1024, (3, 3), (2, 2), padding=1)
        self.bn_b6 = nn.BatchNorm2d(1024)
        self.conv_b6_2 = nn.Conv2d(1024, 1024, (3, 3), (1, 1), padding=1)
        self.conv_b6_3 = nn.Conv2d(512, 1024, (1, 1), (2, 2))
        self.pl = nn.MaxPool2d((2, 2), (1, 1))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        # block1
        x_res = x
        x = self.relu(self.bn_b1(self.conv_b1(x)))
        x = self.bn_b1(self.conv_b1(x))
        x = x + x_res

        # block2
        x_res = self.bn_b2(self.conv_b2_3(x))
        x = self.relu(self.bn_b2(self.conv_b2_1(x)))
        x = self.bn_b2(self.conv_b2_2(x))
        x = x + x_res

        # block3
        x_res = self.bn_b3(self.conv_b3_3(x))
        x = self.relu(self.bn_b3(self.conv_b3_1(x)))
        x = self.bn_b3(self.conv_b3_2(x))
        x = x + x_res

        # block4
        x_res = self.bn_b4(self.conv_b4_3(x))
        x = self.relu(self.bn_b4(self.conv_b4_1(x)))
        x = self.bn_b4(self.conv_b4_2(x))
        x = x + x_res

        # block5
        x_res = self.bn_b5(self.conv_b5_3(x))
        x = self.relu(self.bn_b5(self.conv_b5_1(x)))
        x = self.bn_b5(self.conv_b5_2(x))
        x = x + x_res

        # block6
        x_res = self.bn_b6(self.conv_b6_3(x))
        x = self.relu(self.bn_b6(self.conv_b6_1(x)))
        x = self.bn_b6(self.conv_b6_2(x))
        x = x + x_res
        x = self.pl(x)

        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class p_net(nn.Module):
    def __init__(self, mode):
        super(p_net, self).__init__()
        self.mode = mode
        self.max_pool = nn.MaxPool2d((2, 2), (2, 2))
        # time-domain
        self.conv1_1 = nn.Conv2d(1, 8, (3, 3), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(8, 8, (3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(8, 8, (3, 3), padding=(1, 1))

        self.conv2_1 = nn.Conv2d(8, 16, (3, 3), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(16, 16, (3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(16, 16, (3, 3), padding=(1, 1))
        #fra_domain
        self.conv3_1 = nn.Conv2d(3, 8, (3, 3), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(8, 8, (3, 3), padding=(1, 1))
        self.conv3_3 = nn.Conv2d(8, 8, (3, 3), padding=(1, 1))

        self.conv4_1 = nn.Conv2d(8, 16, (3, 3), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(16, 16, (3, 3), padding=(1, 1))
        self.conv4_3 = nn.Conv2d(16, 16, (3, 3), padding=(1, 1))

        self.conv5_1 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.conv5_2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))

        self.fuse_conv1 = nn.Conv2d(48, 64, (3, 3))
        self.fuse_conv2 = nn.Conv2d(64, 64, (3, 3))
        self.fuse_conv3 = nn.Conv2d(64, 1, (3, 3))

        self.fc1 = nn.Linear(196, 32)
        self.fc2 = nn.Linear(32, 7)

    def forward(self, x_sig, x_fra):
        x_sig = F.relu(self.conv1_1(x_sig))
        x_sig = F.relu(self.conv1_2(x_sig))
        x_sig = F.relu(self.conv1_3(x_sig))
        x_sig = F.relu(self.conv2_1(x_sig))
        x_sig = F.relu(self.conv2_2(x_sig))
        x_sig = self.max_pool(F.relu(self.conv2_3(x_sig)))

        x_fra = F.relu(self.conv3_1(x_fra))
        x_fra = F.relu(self.conv3_2(x_fra))
        x_fra = self.max_pool(F.relu(self.conv3_3(x_fra)))
        x_fra = F.relu(self.conv4_1(x_fra))
        x_fra = F.relu(self.conv4_2(x_fra))
        x_fra = self.max_pool(F.relu(self.conv4_3(x_fra)))
        x_fra = F.relu(self.conv5_1(x_fra))
        x_fra = F.relu(self.conv5_2(x_fra))
        x_fra = self.max_pool(F.relu(self.conv5_2(x_fra)))

        x = torch.cat([x_sig, x_fra], dim=1)
        x = F.relu(self.fuse_conv1(x))
        x = F.relu(self.fuse_conv2(x))
        x = F.relu(self.fuse_conv3(x))
        x = self.fc1(x.view(x.size(0), -1))
        x = F.dropout(x, 0.5, True)
        x = self.fc2(x)
        return x
